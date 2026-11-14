pub mod renderer;
pub mod solver;
pub mod surface_field;

use std::sync::Arc;

pub use renderer::GpuRenderer;
pub use solver::{GpuParticle, GpuSphSolver, GpuUniforms};
pub use surface_field::GpuSurfaceField;

/// Create a standalone Vulkan/wgpu compute+render device and queue for headless CLI tasks.
pub fn create_headless_wgpu_device() -> Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>, String), String>
{
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::PRIMARY,
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| "Failed to find Vulkan / GPU adapter".to_string())?;

    let info = adapter.get_info();
    let device_name = format!("{} ({:?})", info.name, info.backend);

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("LiquidRust Headless GPU Device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: Default::default(),
        },
        None,
    ))
    .map_err(|e| format!("Failed to create GPU device: {}", e))?;

    Ok((Arc::new(device), Arc::new(queue), device_name))
}
