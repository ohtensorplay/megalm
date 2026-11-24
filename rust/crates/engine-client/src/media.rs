//! Sealed memfd creation and SCM_RIGHTS frame transport helpers.
//!
//! The worker validates that every received descriptor is a regular, fully
//! sealed file whose contents hash to the declared SHA-256. We mirror that
//! contract here so a malicious frontend cannot swap media after admission.

use std::ffi::CString;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::raw::{c_int, c_void};

use crate::error::{EngineClientError, Result};

// Not yet exposed by libc on all supported Linux targets.
const MFD_CLOEXEC: libc::c_uint = 0x0001;
const MFD_ALLOW_SEALING: libc::c_uint = 0x0002;

const REQUIRED_SEALS: libc::c_int =
    libc::F_SEAL_SEAL | libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_WRITE;

/// Copy `payload` into a sealed anonymous file and return its owned descriptor.
pub fn seal_memfd(payload: &[u8]) -> Result<OwnedFd> {
    let name = CString::new("megalm-media").unwrap();
    // SAFETY: `name` is NUL-terminated and the syscall takes no owned pointers.
    let raw = unsafe { libc::memfd_create(name.as_ptr(), MFD_CLOEXEC | MFD_ALLOW_SEALING) };
    if raw < 0 {
        return Err(EngineClientError::Media(format!(
            "memfd_create failed: {}",
            io::Error::last_os_error()
        )));
    }
    // SAFETY: `raw` is a fresh, owned file descriptor from a successful call.
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };

    let mut written = 0usize;
    while written < payload.len() {
        // SAFETY: the slice is valid for `payload.len() - written` bytes.
        let count = unsafe {
            libc::write(
                fd.as_raw_fd(),
                payload[written..].as_ptr() as *const c_void,
                payload.len() - written,
            )
        };
        if count <= 0 {
            return Err(EngineClientError::Media(format!(
                "media memfd write made no progress: {}",
                io::Error::last_os_error()
            )));
        }
        written += count as usize;
    }

    // SAFETY: `fd` is a valid memfd created with MFD_ALLOW_SEALING.
    if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, REQUIRED_SEALS) } < 0 {
        return Err(EngineClientError::Media(format!(
            "cannot seal media memfd: {}",
            io::Error::last_os_error()
        )));
    }
    // SAFETY: `fd` is a valid memfd.
    let applied = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GET_SEALS) };
    if applied < 0 || (applied & REQUIRED_SEALS) != REQUIRED_SEALS {
        return Err(EngineClientError::Media(
            "media memfd seals were not applied".into(),
        ));
    }
    Ok(fd)
}

/// Send a frame's bytes with `fds` attached as SCM_RIGHTS ancillary data.
///
/// Returns the number of frame bytes accepted; 0 with ancillary data means the
/// send made no progress. Descriptor rights are attached to the first byte, so
/// a successful partial send must be completed with a plain write.
pub(crate) fn sendmsg_with_fds(fd: RawFd, bytes: &[u8], fds: &[OwnedFd]) -> io::Result<usize> {
    let iov = libc::iovec {
        iov_base: bytes.as_ptr() as *mut c_void,
        iov_len: bytes.len(),
    };
    let fd_values: Vec<c_int> = fds.iter().map(|item| item.as_raw_fd()).collect();
    let data_len = (fd_values.len() * std::mem::size_of::<c_int>()) as libc::c_uint;
    // SAFETY: CMSG_SPACE/LEN are pure compile-time arithmetic on Linux.
    let cmsg_len = unsafe { libc::CMSG_LEN(data_len) };
    let cmsg_space = unsafe { libc::CMSG_SPACE(data_len) } as usize;
    let mut control = vec![0u8; cmsg_space];

    let mut message: libc::msghdr = unsafe { std::mem::zeroed() };
    message.msg_iov = &iov as *const libc::iovec as *mut libc::iovec;
    message.msg_iovlen = 1;
    if !fd_values.is_empty() {
        message.msg_control = control.as_mut_ptr() as *mut c_void;
        message.msg_controllen = cmsg_space;
        // SAFETY: `control` is a zeroed buffer of at least CMSG_SPACE bytes and
        // `message` points at it for the duration of the call.
        unsafe {
            let header = libc::CMSG_FIRSTHDR(&message);
            if header.is_null() {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "no ancillary header available",
                ));
            }
            (*header).cmsg_level = libc::SOL_SOCKET;
            (*header).cmsg_type = libc::SCM_RIGHTS;
            (*header).cmsg_len = cmsg_len as usize;
            std::ptr::copy_nonoverlapping(
                fd_values.as_ptr(),
                libc::CMSG_DATA(header) as *mut c_int,
                fd_values.len(),
            );
        }
    }

    // SAFETY: `message` and all buffers it points to are live for the call.
    let sent = unsafe { libc::sendmsg(fd, &message, 0) };
    if sent < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(sent as usize)
    }
}

/// Duplicate a sealed descriptor so several choices can reference the same
/// immutable media without rewriting the payload.
pub fn duplicate_fd(fd: &OwnedFd) -> Result<OwnedFd> {
    // SAFETY: `fd` is a valid owned descriptor for the call's duration.
    let raw = unsafe { libc::dup(fd.as_raw_fd()) };
    if raw < 0 {
        return Err(EngineClientError::Media(format!(
            "cannot duplicate media descriptor: {}",
            io::Error::last_os_error()
        )));
    }
    // SAFETY: `raw` is a fresh, owned descriptor from a successful dup.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(align(8))]
    struct Control([u8; 256]);

    #[test]
    fn scm_rights_carries_sealed_fd() {
        use std::os::unix::net::UnixStream;

        let (sender, receiver) = UnixStream::pair().unwrap();
        let sealed = seal_memfd(b"media-bytes").unwrap();
        let frame = b"HEADERBODY";
        let sent = sendmsg_with_fds(
            sender.as_raw_fd(),
            frame,
            std::slice::from_ref(&sealed),
        )
        .unwrap();
        assert_eq!(sent, frame.len());

        let mut buffer = [0u8; 16];
        let iov = libc::iovec {
            iov_base: buffer.as_mut_ptr() as *mut c_void,
            iov_len: buffer.len(),
        };
        let mut control = Control([0u8; 256]);
        let mut message: libc::msghdr = unsafe { std::mem::zeroed() };
        message.msg_iov = &iov as *const libc::iovec as *mut libc::iovec;
        message.msg_iovlen = 1;
        message.msg_control = control.0.as_mut_ptr() as *mut c_void;
        message.msg_controllen = control.0.len();
        // SAFETY: all buffers are live and correctly sized for the call.
        let received = unsafe { libc::recvmsg(receiver.as_raw_fd(), &mut message, 0) };
        assert!(received > 0);
        assert_eq!(&buffer[..received as usize], frame);

        // SAFETY: `message` was filled by a successful recvmsg with a control
        // buffer large enough for one SCM_RIGHTS message.
        let fd = unsafe {
            let header = libc::CMSG_FIRSTHDR(&message);
            assert!(!header.is_null());
            assert_eq!((*header).cmsg_level, libc::SOL_SOCKET);
            assert_eq!((*header).cmsg_type, libc::SCM_RIGHTS);
            let data = libc::CMSG_DATA(header) as *const c_int;
            OwnedFd::from_raw_fd(*data)
        };
        let mut payload = [0u8; 11];
        // SAFETY: `fd` is the transferred, owned descriptor.
        let read = unsafe {
            libc::pread(
                fd.as_raw_fd(),
                payload.as_mut_ptr() as *mut c_void,
                payload.len(),
                0,
            )
        };
        assert_eq!(read, 11);
        assert_eq!(&payload, b"media-bytes");
    }

    #[test]
    fn sealed_memfd_has_contents_and_all_seals() {
        let payload = b"hello media";
        let fd = seal_memfd(payload).unwrap();
        let mut buffer = [0u8; 11];
        // SAFETY: `fd` is a valid sealed memfd and the buffer has 11 bytes.
        let read = unsafe {
            libc::pread(
                fd.as_raw_fd(),
                buffer.as_mut_ptr() as *mut c_void,
                buffer.len(),
                0,
            )
        };
        assert_eq!(read, payload.len() as libc::ssize_t);
        assert_eq!(&buffer, payload);
        // SAFETY: `fd` is a valid memfd.
        let seals = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GET_SEALS) };
        assert_eq!(seals & REQUIRED_SEALS, REQUIRED_SEALS);
    }
}
