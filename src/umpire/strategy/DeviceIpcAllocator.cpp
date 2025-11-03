//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/DeviceIpcAllocator.hpp"

#include <sstream>

#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/resource/HostSharedMemoryResource.hpp"
#include "umpire/strategy/NamedAllocationStrategy.hpp"
#include "umpire/util/MPI.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"
#include "umpire/util/error.hpp"

#if defined(UMPIRE_ENABLE_CUDA)
#define gpuIpcGetMemHandle cudaIpcGetMemHandle
#define gpuIpcCloseMemHandle cudaIpcCloseMemHandle
#define gpuIpcOpenMemHandle cudaIpcOpenMemHandle
#define gpuGetDevice cudaGetDevice
#define gpuSetDevice cudaSetDevice
#define gpuError cudaError_t
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuGetErrorString cudaGetErrorString
#define gpuSuccess cudaSuccess
#define gpuDeviceProp cudaDeviceProp
#define gpuIpcMemLazyEnablePeerAccess cudaIpcMemLazyEnablePeerAccess
#elif defined(UMPIRE_ENABLE_HIP)
#define gpuIpcGetMemHandle hipIpcGetMemHandle
#define gpuIpcCloseMemHandle hipIpcCloseMemHandle
#define gpuIpcOpenMemHandle hipIpcOpenMemHandle
#define gpuGetDevice hipGetDevice
#define gpuSetDevice hipSetDevice
#define gpuError hipError_t
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuGetErrorString hipGetErrorString
#define gpuSuccess hipSuccess
#define gpuDeviceProp hipDeviceProp_t
#define gpuIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess
#endif

namespace umpire {
namespace strategy {

DeviceIpcAllocator::DeviceIpcAllocator(const std::string& name, int id) noexcept
    : DeviceIpcAllocator(name, id, umpire::ResourceManager::getInstance().getAllocator("DEVICE"),
                         MemoryResourceTraits::shared_scope::socket)
{
}

DeviceIpcAllocator::DeviceIpcAllocator(const std::string& name, int id, Allocator device_allocator,
                                       MemoryResourceTraits::shared_scope scope,
                                       std::size_t shared_memory_size) noexcept
    : AllocationStrategy(name, id, device_allocator.getAllocationStrategy(), "DeviceIpcAllocator"),
      m_device_allocator(device_allocator.getAllocationStrategy()),
      m_scope_rank(0),
      m_is_scope_leader(false),
      m_scope_color(MPI_COMM_TYPE_SHARED)
{
  setup_shared_scope(scope);

  auto& rm = umpire::ResourceManager::getInstance();
  auto traits = umpire::get_default_resource_traits("SHARED");
  traits.size = shared_memory_size;
  auto shared_alloc = rm.makeResource("SHARED::" + name + std::to_string(m_scope_color), traits);
  m_shared_allocator = shared_alloc.getAllocationStrategy();
  m_shared_memory_resource = dynamic_cast<umpire::resource::HostSharedMemoryResource*>(m_shared_allocator);
}

DeviceIpcAllocator::~DeviceIpcAllocator()
{
  MPI_Barrier(m_scope_comm);

  if (m_is_scope_leader) {
    for (auto& item : m_allocation_names) {
      try {
        m_device_allocator->deallocate_internal(item.first);
      } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "Exception during cleanup: " << e.what();
        UMPIRE_LOG(Warning, oss.str());
      }
    }
    m_allocation_names.clear();
  }

  MPI_Comm_free(&m_scope_comm);
}

void DeviceIpcAllocator::setup_shared_scope(MemoryResourceTraits::shared_scope scope)
{
  if (scope == MemoryResourceTraits::shared_scope::node) {
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &m_scope_comm);
    MPI_Comm_rank(m_scope_comm, &m_scope_rank);
    {
      std::ostringstream oss;
      oss << "Node scope: rank " << m_scope_rank << " in node communicator";
      UMPIRE_LOG(Debug, oss.str());
    }
  } else if (scope == MemoryResourceTraits::shared_scope::socket) {
    // Get current device
    int device_id;
    gpuError err = gpuGetDevice(&device_id);
    if (err != gpuSuccess) {
      std::ostringstream oss;
      oss << "gpuGetDevice failed with error: " << gpuGetErrorString(err);
      UMPIRE_ERROR(runtime_error, oss.str());
    }

    // Get device properties
    gpuDeviceProp props;
    err = gpuGetDeviceProperties(&props, device_id);
    if (err != gpuSuccess) {
      std::ostringstream oss;
      oss << "gpuGetDeviceProperties failed with error: " << gpuGetErrorString(err);
      UMPIRE_ERROR(runtime_error, oss.str());
    }

    // Use PCI domain, bus, and device as color for MPI communicator split
    m_scope_color = props.pciDomainID * 1000000 + props.pciBusID * 1000 + props.pciDeviceID;

    {
      std::ostringstream oss;
      oss << "Socket scope: using color " << m_scope_color << " for PCI split";
      UMPIRE_LOG(Debug, oss.str());
    }
    MPI_Comm_split(MPI_COMM_WORLD, m_scope_color, 0, &m_scope_comm);
    MPI_Comm_rank(m_scope_comm, &m_scope_rank);
    {
      std::ostringstream oss;
      oss << "Socket scope: rank " << m_scope_rank << " in socket communicator";
      UMPIRE_LOG(Debug, oss.str());
    }
  } else {
    UMPIRE_ERROR(runtime_error, "Unsupported scope for DeviceIpcAllocator");
  }

  m_is_scope_leader = (m_scope_rank == 0);
  {
    std::ostringstream oss;
    oss << "Rank " << m_scope_rank << " is " << (m_is_scope_leader ? "scope" : "not scope") << " leader";
    UMPIRE_LOG(Debug, oss.str());
  }
}

std::string DeviceIpcAllocator::generate_allocation_name(std::size_t size_in_bytes)
{
  static unsigned long counter = 0;
  std::stringstream ss;
  ss << m_name << "_" << m_scope_color << "_" << size_in_bytes << "_" << counter++;
  return ss.str();
}

void* DeviceIpcAllocator::allocate(std::size_t bytes)
{
  {
    std::ostringstream oss;
    oss << "(size_in_bytes=" << bytes << ")";
    UMPIRE_LOG(Debug, oss.str());
  }

  std::string allocation_name = generate_allocation_name(bytes);

  void* ptr = nullptr;

  if (m_is_scope_leader) {
    ptr = create(allocation_name, bytes);
  } else {
    ptr = import(allocation_name);
  }

  if (!ptr) {
    UMPIRE_ERROR(runtime_error, "Failed to allocate/import device memory");
  }

  // Track allocation name (only in leader process)
  if (m_is_scope_leader) {
    m_allocation_names[ptr] = allocation_name;
  }

  return ptr;
}

void* DeviceIpcAllocator::create(const std::string& name, std::size_t size_in_bytes)
{
  void* ptr = m_device_allocator->allocate_internal(size_in_bytes);
  {
    std::ostringstream oss;
    oss << "Leader allocated device memory at " << ptr << " (size: " << size_in_bytes << ")";
    UMPIRE_LOG(Debug, oss.str());
  }

  IpcHandleInfo* handle_info = create_handle_info(name, size_in_bytes);
  if (!handle_info) {
    m_device_allocator->deallocate_internal(ptr);
    UMPIRE_ERROR(runtime_error, "Failed to create handle info in shared memory");
  }

  auto err = gpuIpcGetMemHandle(&handle_info->handle, ptr);
  if (err != gpuSuccess) {
    m_device_allocator->deallocate_internal(ptr);
    m_shared_allocator->deallocate_internal(handle_info);
    std::ostringstream oss;
    oss << "gpuIpcGetMemHandle failed with error: " << gpuGetErrorString(err);
    UMPIRE_ERROR(runtime_error, oss.str());
  }

  // Setup handle info fields
  handle_info->size = size_in_bytes;
  err = gpuGetDevice(&handle_info->device_id);
  if (err != gpuSuccess) {
    std::ostringstream oss;
    oss << "gpuGetDevice failed with error: " << gpuGetErrorString(err);
    UMPIRE_ERROR(runtime_error, oss.str());
  }
  handle_info->is_initialized.store(true, std::memory_order_release);

  // Signal followers that the handle is ready
  MPI_Barrier(m_scope_comm);
  {
    std::ostringstream oss;
    oss << "Leader completed IPC setup for device memory at " << ptr;
    UMPIRE_LOG(Debug, oss.str());
  }

  return ptr;
}

void* DeviceIpcAllocator::import(const std::string& name)
{
  // Wait for leader to initialize the handle
  MPI_Barrier(m_scope_comm);

  IpcHandleInfo* handle_info = get_handle_info(name);
  if (!handle_info || !handle_info->is_initialized.load(std::memory_order_acquire)) {
    UMPIRE_ERROR(runtime_error, "Failed to get initialized IPC handle");
    return nullptr;
  }

  int current_device, target_device = handle_info->device_id;
  gpuError err = gpuGetDevice(&current_device);
  if (err != gpuSuccess) {
    std::ostringstream oss;
    oss << "gpuGetDevice failed with error: " << gpuGetErrorString(err);
    UMPIRE_ERROR(runtime_error, oss.str());
  }
  bool device_switched = false;
  if (current_device != target_device) {
    err = gpuSetDevice(target_device);
    if (err != gpuSuccess) {
      std::ostringstream oss;
      oss << "gpuSetDevice failed with error: " << gpuGetErrorString(err);
      UMPIRE_ERROR(runtime_error, oss.str());
    }
    device_switched = true;
  }

  void* ptr = nullptr;
  err = gpuIpcOpenMemHandle(&ptr, handle_info->handle, gpuIpcMemLazyEnablePeerAccess);
  {
    std::ostringstream oss;
    oss << "Follower opened IPC handle to device memory at " << ptr;
    UMPIRE_LOG(Debug, oss.str());
  }
  if (err != gpuSuccess) {
    auto store_error_temp = gpuGetErrorString(err);
    if (device_switched) {
      err = gpuSetDevice(current_device);
      if (err != gpuSuccess) {
        std::ostringstream oss;
        oss << "gpuSetDevice failed with error: " << gpuGetErrorString(err);
        UMPIRE_ERROR(runtime_error, oss.str());
      }
    }
    {
      std::ostringstream oss;
      oss << "gpuIpcOpenMemHandle failed with error: " << store_error_temp;
      UMPIRE_ERROR(runtime_error, oss.str());
    }
    return nullptr;
  }

  if (device_switched) {
    err = gpuSetDevice(current_device);
    if (err != gpuSuccess) {
      std::ostringstream oss;
      oss << "gpuSetDevice failed with error: " << gpuGetErrorString(err);
      UMPIRE_ERROR(runtime_error, oss.str());
    }
  }

  {
    std::ostringstream oss;
    oss << "Follower successfully imported device memory at " << ptr;
    UMPIRE_LOG(Debug, oss.str());
  }
  return ptr;
}

void DeviceIpcAllocator::deallocate(void* ptr, std::size_t)
{
  {
    std::ostringstream oss;
    oss << "(ptr=" << ptr << ")";
    UMPIRE_LOG(Debug, oss.str());
  }

  MPI_Barrier(m_scope_comm);

  if (m_is_scope_leader) {
    auto it = m_allocation_names.find(ptr);
    if (it == m_allocation_names.end()) {
      UMPIRE_ERROR(runtime_error, "Cannot deallocate unknown pointer");
    }

    const std::string& allocation_name = it->second;

    m_device_allocator->deallocate_internal(ptr);

    IpcHandleInfo* handle_info = get_handle_info(allocation_name);
    if (handle_info) {
      m_shared_allocator->deallocate_internal(handle_info);
    }

    m_allocation_names.erase(it);
  } else {
    gpuError err = gpuIpcCloseMemHandle(ptr);
    if (err != gpuSuccess) {
      std::ostringstream oss;
      oss << "gpuIpcCloseMemHandle failed with error: " << gpuGetErrorString(err);
      UMPIRE_ERROR(runtime_error, oss.str());
    }
  }
}

DeviceIpcAllocator::IpcHandleInfo* DeviceIpcAllocator::get_handle_info(const std::string& name)
{
  void* shared_ptr = m_shared_memory_resource->find_pointer_from_name(name);
  {
    std::ostringstream oss;
    oss << "Found shared memory at " << shared_ptr << " for " << name;
    UMPIRE_LOG(Debug, oss.str());
  }
  return static_cast<IpcHandleInfo*>(shared_ptr);
}

DeviceIpcAllocator::IpcHandleInfo* DeviceIpcAllocator::create_handle_info(const std::string& name,
                                                                          std::size_t size_in_bytes)
{
  if (m_is_scope_leader) {
    void* shared_ptr = m_shared_memory_resource->allocate_named_internal(name, sizeof(IpcHandleInfo));
    {
      std::ostringstream oss;
      oss << "Leader created shared memory at " << shared_ptr << " for " << name;
      UMPIRE_LOG(Debug, oss.str());
    }

    if (shared_ptr) {
      IpcHandleInfo* info = static_cast<IpcHandleInfo*>(shared_ptr);
      info->size = size_in_bytes;
      info->device_id = 0;
      info->is_initialized.store(false, std::memory_order_relaxed);
      return info;
    }
  }

  return nullptr;
}

MPI_Comm DeviceIpcAllocator::get_scope_communicator()
{
  return m_scope_comm;
}

Platform DeviceIpcAllocator::getPlatform() noexcept
{
  return m_device_allocator->getPlatform();
}

MemoryResourceTraits DeviceIpcAllocator::getTraits() const noexcept
{
  return m_device_allocator->getTraits();
}

} // end of namespace strategy
} // end of namespace umpire

