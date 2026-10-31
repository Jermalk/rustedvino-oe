// ============================================================
// src/gpu_driver.rs — which GPU driver (OpenCL runtime) is this process using (step 8)
// ============================================================
// OpenVINO's GPU plugin exposes no driver-version property, and RustedVINO does not ship a GPU
// driver: it runs on whatever the machine has (Intel's apt repo, the distribution, a newer manual
// install). Knowing which one is the first question for any GPU problem report (the 2026-10-27 B60
// exit fault behaved differently on NEO 25.18 and 26.35), so it is read here from OpenCL itself —
// `CL_DRIVER_VERSION` plus the PCI address that ties an OpenCL device to OpenVINO's GPU.N — and
// shown in the startup log, the admin server block and `rv status`. Read-only: no OpenCL context.
//
// The system OpenCL loader is opened with `dlopen` (Unix only) instead of linking it: a box without
// OpenCL still boots, and no new crate is needed. Windows: not read yet (empty list).
// ============================================================

/// One `OpenCL` GPU device as the installed driver reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OclGpu {
    /// `CL_DEVICE_NAME`.
    pub name: String,
    /// `CL_DRIVER_VERSION` (for Intel's compute runtime: the NEO version, e.g. `25.18.33578`).
    pub driver_version: String,
    /// PCI address `dddd:bb:dd.f` (`cl_khr_pci_bus_info`), when the driver reports it.
    pub pci: Option<String>,
}

/// `dddd:bb:dd.f`, the form DRM fdinfo and `DeviceInfo::pdev` use.
#[must_use]
pub fn format_pci(domain: u32, bus: u32, device: u32, function: u32) -> String {
    format!("{domain:04x}:{bus:02x}:{device:02x}.{function:x}")
}

/// Every `OpenCL` GPU device on every platform; empty when `OpenCL` is absent or unreadable.
#[must_use]
pub fn opencl_gpus() -> Vec<OclGpu> {
    #[cfg(unix)]
    {
        unix::query().unwrap_or_default()
    }
    #[cfg(not(unix))]
    {
        Vec::new()
    }
}

/// The running kernel's release (`/proc/sys/kernel/osrelease`), Linux only.
#[must_use]
pub fn kernel_release() -> Option<String> {
    std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}

/// The kernel driver bound to the PCI device `pci` (`dddd:bb:dd.f`): the name of its
/// `/sys/bus/pci/devices/<pci>/driver` link, e.g. `xe` or `i915`. `None` off Linux or when unbound.
#[must_use]
pub fn kernel_driver(pci: &str) -> Option<String> {
    std::fs::read_link(format!("/sys/bus/pci/devices/{pci}/driver"))
        .ok()?
        .file_name()?
        .to_str()
        .map(str::to_owned)
}

/// What an operator can do about a model load past its deadline that may never return (D-04 S9,
/// the ABU-05 class). On an Arc Pro B60 the xe driver stayed stuck for up to 3 h 37 min after a
/// process was killed in the middle of GPU work; a GT reset freed it in milliseconds, after which
/// the server committed the late load and carried on (verified 2026-10-29, kernel 7.0, one
/// resident model). So the reset command is named only where it was verified: a discrete GPU on
/// the xe driver with a known PCI address. Other GPUs and the NPU get "reboot"; a CPU load holds
/// no device-driver state, so a server restart releases it.
#[must_use]
pub fn stalled_load_recovery(
    kind: Option<crate::device_inventory::DeviceKind>,
    pci: Option<&str>,
    driver: Option<&str>,
) -> String {
    use crate::device_inventory::DeviceKind;
    match (kind, pci, driver) {
        (Some(DeviceKind::DiscreteGpu), Some(pci), Some("xe")) => format!(
            "if it never returns, the GPU driver may be stuck (a known xe issue): a GPU reset \
             cleared it on an Arc Pro B60 without a server restart - `sudo cat \
             /sys/kernel/debug/dri/{pci}/gt0/force_reset_sync` (kernel 7.0 path); models already \
             loaded on this GPU may need reloading after it. Otherwise reboot; restarting the \
             server is not a reliable fix"
        ),
        (Some(DeviceKind::Cpu), ..) => "if it never returns, restart the server (`rv restart`): \
             a CPU load holds no device-driver state; the stop waits up to 600 s for the load \
             first"
            .to_owned(),
        (Some(DeviceKind::Npu), ..) => "if it never returns, the NPU driver may be stuck: reboot \
             the machine (restarting the server is not a reliable fix)"
            .to_owned(),
        (Some(DeviceKind::DiscreteGpu | DeviceKind::IntegratedGpu), ..) => {
            "if it never returns, the GPU driver may be stuck: reboot the machine (restarting \
             the server is not a reliable fix)"
                .to_owned()
        }
        _ => "if it never returns, reboot the machine".to_owned(),
    }
}

#[cfg(unix)]
mod unix {
    use super::{OclGpu, format_pci};
    use std::ffi::{CStr, c_void};

    type ClInt = i32;
    type ClUint = u32;
    type ClPlatformId = *mut c_void;
    type ClDeviceId = *mut c_void;
    type GetPlatformIds = unsafe extern "C" fn(ClUint, *mut ClPlatformId, *mut ClUint) -> ClInt;
    type GetDeviceIds =
        unsafe extern "C" fn(ClPlatformId, u64, ClUint, *mut ClDeviceId, *mut ClUint) -> ClInt;
    type GetDeviceInfo =
        unsafe extern "C" fn(ClDeviceId, ClUint, usize, *mut c_void, *mut usize) -> ClInt;

    const CL_SUCCESS: ClInt = 0;
    const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
    const CL_DEVICE_NAME: ClUint = 0x102B;
    const CL_DRIVER_VERSION: ClUint = 0x102D;
    const CL_DEVICE_PCI_BUS_INFO_KHR: ClUint = 0x410F;

    /// Look up `name` in `lib` as a function pointer of type `T`.
    ///
    /// # Safety
    /// `T` must be the exact C signature of the symbol.
    unsafe fn sym<T: Copy>(lib: *mut c_void, name: &CStr) -> Option<T> {
        // SAFETY: `lib` is a live handle from `dlopen`; `name` is NUL-terminated.
        let p = unsafe { libc::dlsym(lib, name.as_ptr()) };
        if p.is_null() {
            return None;
        }
        // SAFETY: the caller guarantees `T` is the symbol's function-pointer type, which has the
        // size of a pointer; `p` is non-null.
        Some(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&p) })
    }

    /// A string device property.
    ///
    /// # Safety
    /// `get` must be `clGetDeviceInfo` and `dev` a device id it returned.
    unsafe fn string_info(get: GetDeviceInfo, dev: ClDeviceId, param: ClUint) -> Option<String> {
        let mut buf = [0u8; 256];
        let mut len = 0usize;
        // SAFETY: `buf` is writable for its full length, which is passed as the size limit.
        let rc = unsafe { get(dev, param, buf.len(), buf.as_mut_ptr().cast(), &raw mut len) };
        if rc != CL_SUCCESS {
            return None;
        }
        let end = buf
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(len.min(buf.len()));
        Some(String::from_utf8_lossy(&buf[..end]).trim().to_owned())
    }

    pub(super) fn query() -> Option<Vec<OclGpu>> {
        // SAFETY: `dlopen` with a constant NUL-terminated name; a null handle is handled. The
        // handle is deliberately never closed: OpenVINO's GPU plugin keeps the same loader open.
        let lib = unsafe {
            libc::dlopen(
                c"libOpenCL.so.1".as_ptr(),
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            )
        };
        if lib.is_null() {
            return None;
        }
        // SAFETY: the three types match the OpenCL 1.2 C prototypes of these symbols.
        let (get_platforms, get_devices, get_info) = unsafe {
            (
                sym::<GetPlatformIds>(lib, c"clGetPlatformIDs")?,
                sym::<GetDeviceIds>(lib, c"clGetDeviceIDs")?,
                sym::<GetDeviceInfo>(lib, c"clGetDeviceInfo")?,
            )
        };
        let mut platforms = [std::ptr::null_mut::<c_void>(); 16];
        let mut n_platforms: ClUint = 0;
        // SAFETY: the array holds 16 entries and 16 is passed as its capacity.
        if unsafe { get_platforms(16, platforms.as_mut_ptr(), &raw mut n_platforms) } != CL_SUCCESS
        {
            return None;
        }
        let mut out = Vec::new();
        for &platform in platforms.iter().take(n_platforms.min(16) as usize) {
            let mut devices = [std::ptr::null_mut::<c_void>(); 16];
            let mut n_devices: ClUint = 0;
            // SAFETY: `platform` came from `clGetPlatformIDs`; the array capacity is passed.
            let rc = unsafe {
                get_devices(
                    platform,
                    CL_DEVICE_TYPE_GPU,
                    16,
                    devices.as_mut_ptr(),
                    &raw mut n_devices,
                )
            };
            if rc != CL_SUCCESS {
                continue;
            }
            for &dev in devices.iter().take(n_devices.min(16) as usize) {
                // SAFETY: `get_info` is clGetDeviceInfo and `dev` came from clGetDeviceIDs.
                let name = unsafe { string_info(get_info, dev, CL_DEVICE_NAME) };
                // SAFETY: as above.
                let driver_version = unsafe { string_info(get_info, dev, CL_DRIVER_VERSION) };
                let mut pci = [0u32; 4];
                // SAFETY: `cl_device_pci_bus_info_khr` is four cl_uint (16 bytes) = `pci`.
                let rc = unsafe {
                    get_info(
                        dev,
                        CL_DEVICE_PCI_BUS_INFO_KHR,
                        std::mem::size_of_val(&pci),
                        pci.as_mut_ptr().cast(),
                        std::ptr::null_mut(),
                    )
                };
                if let (Some(name), Some(driver_version)) = (name, driver_version) {
                    out.push(OclGpu {
                        name,
                        driver_version,
                        pci: (rc == CL_SUCCESS).then(|| format_pci(pci[0], pci[1], pci[2], pci[3])),
                    });
                }
            }
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pci_address_matches_the_drm_fdinfo_form() {
        assert_eq!(format_pci(0, 3, 0, 0), "0000:03:00.0");
        assert_eq!(format_pci(0, 0, 2, 0), "0000:00:02.0");
        assert_eq!(format_pci(0x10, 0xab, 0x1f, 7), "0010:ab:1f.7");
    }

    /// Smoke test (no GPU access in unit tests: `opencl_gpus` is exercised live).
    #[test]
    fn kernel_release_never_panics() {
        let _ = kernel_release();
    }

    /// D-04 S9: the reset command only for a discrete xe GPU with a PCI address; every other
    /// device gets advice that fits it (review S9 #2, #3).
    #[test]
    fn stalled_load_recovery_fits_the_device() {
        use crate::device_inventory::DeviceKind::{Cpu, DiscreteGpu, IntegratedGpu, Npu, Other};
        let pci = Some("0000:03:00.0");
        let xe = stalled_load_recovery(Some(DiscreteGpu), pci, Some("xe"));
        assert!(xe.contains("sudo cat /sys/kernel/debug/dri/0000:03:00.0/gt0/force_reset_sync"));
        assert!(xe.contains("may need reloading"), "{xe}");
        for (kind, pci, driver) in [
            (Some(DiscreteGpu), pci, Some("i915")),
            (Some(DiscreteGpu), None, Some("xe")),
            (Some(DiscreteGpu), pci, None),
            (Some(IntegratedGpu), pci, Some("xe")),
        ] {
            let gpu = stalled_load_recovery(kind, pci, driver);
            assert!(!gpu.contains("force_reset"), "{gpu}");
            assert!(
                gpu.contains("GPU driver") && gpu.contains("reboot"),
                "{gpu}"
            );
        }
        let cpu = stalled_load_recovery(Some(Cpu), None, None);
        assert!(
            cpu.contains("restart the server") && !cpu.contains("GPU"),
            "{cpu}"
        );
        let npu = stalled_load_recovery(Some(Npu), None, None);
        assert!(npu.contains("NPU driver") && !npu.contains("GPU"), "{npu}");
        for kind in [Some(Other), None] {
            let other = stalled_load_recovery(kind, None, None);
            assert!(
                other.contains("reboot") && !other.contains("GPU"),
                "{other}"
            );
        }
    }

    /// Smoke test: an address that cannot exist has no driver.
    #[test]
    fn kernel_driver_of_a_missing_device_is_none() {
        assert_eq!(kernel_driver("ffff:ff:ff.7"), None);
    }
}
