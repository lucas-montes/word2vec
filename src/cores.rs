
use std::mem;

use libc::{cpu_set_t, sched_getaffinity, sched_setaffinity, CPU_ISSET, CPU_SET, CPU_SETSIZE};

/// This represents a CPU core.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CoreId {
    pub id: usize,
}

pub fn get_core_ids() -> Option<Vec<CoreId>> {
    if let Some(full_set) = get_affinity_mask() {
        let mut core_ids: Vec<CoreId> = Vec::new();

        for i in 0..CPU_SETSIZE as usize {
            if unsafe { CPU_ISSET(i, &full_set) } {
                core_ids.push(CoreId { id: i });
            }
        }

        Some(core_ids)
    } else {
        None
    }
}

pub fn set_for_current(core_id: CoreId) -> bool {
    // Turn `core_id` into a `libc::cpu_set_t` with only
    // one core active.
    let mut set = new_cpu_set();

    unsafe { CPU_SET(core_id.id, &mut set) };

    // Set the current thread's core affinity.
    let res = unsafe {
        sched_setaffinity(
            0, // Defaults to current thread
            mem::size_of::<cpu_set_t>(),
            &set,
        )
    };
    res == 0
}

fn get_affinity_mask() -> Option<cpu_set_t> {
    let mut set = new_cpu_set();

    // Try to get current core affinity mask.
    let result = unsafe {
        sched_getaffinity(
            0, // Defaults to current thread
            mem::size_of::<cpu_set_t>(),
            &mut set,
        )
    };

    if result == 0 {
        Some(set)
    } else {
        None
    }
}

fn new_cpu_set() -> cpu_set_t {
    unsafe { mem::zeroed::<cpu_set_t>() }
}
