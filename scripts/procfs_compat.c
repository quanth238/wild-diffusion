#define _GNU_SOURCE

#include <ctype.h>
#include <dlfcn.h>
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <link.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <unistd.h>

/*
 * Slurm occasionally drops GPU steps into a mount namespace where /proc exists
 * only as an empty directory. CUDA then fails to initialize because a small
 * procfs surface is missing. This preload shim redirects those paths to local
 * stand-ins that are good enough for torch + CUDA to proceed.
 */

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static __thread int g_in_hook = 0;

static int (*g_real_open)(const char *pathname, int flags, ...) = NULL;
static int (*g_real_open64)(const char *pathname, int flags, ...) = NULL;
static int (*g_real_openat)(int dirfd, const char *pathname, int flags, ...) = NULL;
static int (*g_real_openat64)(int dirfd, const char *pathname, int flags, ...) = NULL;
static FILE *(*g_real_fopen)(const char *pathname, const char *mode) = NULL;
static FILE *(*g_real_fopen64)(const char *pathname, const char *mode) = NULL;
static int (*g_real_access)(const char *pathname, int mode) = NULL;
static int (*g_real_faccessat)(int dirfd, const char *pathname, int mode, int flags) = NULL;
static int (*g_real_stat)(const char *pathname, struct stat *buf) = NULL;
static int (*g_real_lstat)(const char *pathname, struct stat *buf) = NULL;
static int (*g_real_stat64)(const char *pathname, struct stat64 *buf) = NULL;
static int (*g_real_lstat64)(const char *pathname, struct stat64 *buf) = NULL;
static int (*g_real___xstat)(int ver, const char *pathname, struct stat *buf) = NULL;
static int (*g_real___lxstat)(int ver, const char *pathname, struct stat *buf) = NULL;
static int (*g_real___xstat64)(int ver, const char *pathname, struct stat64 *buf) = NULL;
static int (*g_real___lxstat64)(int ver, const char *pathname, struct stat64 *buf) = NULL;
static int (*g_real_newfstatat)(int dirfd, const char *pathname, struct stat *buf, int flags) = NULL;
static int (*g_real_statx)(int dirfd, const char *pathname, int flags, unsigned int mask, struct statx *buf) = NULL;

static char g_root_dir[PATH_MAX];
static char g_self_dir[PATH_MAX];
static char g_task_dir[PATH_MAX];
static char g_sys_vm_dir[PATH_MAX];
static char g_cpuinfo_path[PATH_MAX];
static char g_maps_path[PATH_MAX];
static char g_mmap_min_addr_path[PATH_MAX];
static char g_cgroup_path[PATH_MAX];
static char g_meminfo_path[PATH_MAX];
static char g_comm_path[PATH_MAX];

static int mkdir_p(const char *path, mode_t mode) {
    char tmp[PATH_MAX];
    size_t len;
    char *p;

    if (path == NULL || path[0] == '\0') {
        errno = EINVAL;
        return -1;
    }
    len = strnlen(path, sizeof(tmp));
    if (len == 0 || len >= sizeof(tmp)) {
        errno = ENAMETOOLONG;
        return -1;
    }
    memcpy(tmp, path, len + 1);
    if (tmp[len - 1] == '/') {
        tmp[len - 1] = '\0';
    }
    for (p = tmp + 1; *p != '\0'; ++p) {
        if (*p != '/') {
            continue;
        }
        *p = '\0';
        if (mkdir(tmp, mode) != 0 && errno != EEXIST) {
            return -1;
        }
        *p = '/';
    }
    if (mkdir(tmp, mode) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

static void resolve_symbols(void) {
    g_real_open = dlsym(RTLD_NEXT, "open");
    g_real_open64 = dlsym(RTLD_NEXT, "open64");
    g_real_openat = dlsym(RTLD_NEXT, "openat");
    g_real_openat64 = dlsym(RTLD_NEXT, "openat64");
    g_real_fopen = dlsym(RTLD_NEXT, "fopen");
    g_real_fopen64 = dlsym(RTLD_NEXT, "fopen64");
    g_real_access = dlsym(RTLD_NEXT, "access");
    g_real_faccessat = dlsym(RTLD_NEXT, "faccessat");
    g_real_stat = dlsym(RTLD_NEXT, "stat");
    g_real_lstat = dlsym(RTLD_NEXT, "lstat");
    g_real_stat64 = dlsym(RTLD_NEXT, "stat64");
    g_real_lstat64 = dlsym(RTLD_NEXT, "lstat64");
    g_real___xstat = dlsym(RTLD_NEXT, "__xstat");
    g_real___lxstat = dlsym(RTLD_NEXT, "__lxstat");
    g_real___xstat64 = dlsym(RTLD_NEXT, "__xstat64");
    g_real___lxstat64 = dlsym(RTLD_NEXT, "__lxstat64");
    g_real_newfstatat = dlsym(RTLD_NEXT, "newfstatat");
    g_real_statx = dlsym(RTLD_NEXT, "statx");
}

static int write_text_file(const char *path, const char *contents) {
    int fd;
    size_t len = strlen(contents);

    fd = syscall(SYS_openat, AT_FDCWD, path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) {
        return -1;
    }
    if (len > 0 && write(fd, contents, len) < 0) {
        int err = errno;
        close(fd);
        errno = err;
        return -1;
    }
    close(fd);
    return 0;
}

static void build_cpuinfo_fallback(char *buf, size_t buf_size) {
    struct utsname uts;
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    size_t used = 0;
    long i;

    if (nproc < 1) {
        nproc = 1;
    }
    if (uname(&uts) != 0) {
        memset(&uts, 0, sizeof(uts));
        strncpy(uts.machine, "x86_64", sizeof(uts.machine) - 1);
    }

    for (i = 0; i < nproc; ++i) {
        int written = snprintf(
            buf + used,
            buf_size - used,
            "processor\t: %ld\n"
            "vendor_id\t: GenuineIntel\n"
            "cpu family\t: 6\n"
            "model\t\t: 143\n"
            "model name\t: procfs-compat %s\n"
            "stepping\t: 0\n"
            "microcode\t: 0x0\n"
            "cpu MHz\t\t: 0.000\n"
            "cache size\t: 0 KB\n"
            "physical id\t: 0\n"
            "siblings\t: %ld\n"
            "core id\t\t: %ld\n"
            "cpu cores\t: %ld\n"
            "apicid\t\t: %ld\n"
            "initial apicid\t: %ld\n"
            "fpu\t\t: yes\n"
            "fpu_exception\t: yes\n"
            "cpuid level\t: 0\n"
            "wp\t\t: yes\n"
            "flags\t\t: fpu sse sse2 avx avx2 avx512f\n"
            "bugs\t\t:\n"
            "bogomips\t: 0.00\n"
            "clflush size\t: 64\n"
            "cache_alignment\t: 64\n"
            "address sizes\t: 46 bits physical, 48 bits virtual\n"
            "power management:\n\n",
            i,
            uts.machine,
            nproc,
            i,
            nproc,
            i,
            i
        );
        if (written < 0 || (size_t) written >= buf_size - used) {
            break;
        }
        used += (size_t) written;
    }
    if (used == 0 && buf_size > 1) {
        snprintf(buf, buf_size, "processor\t: 0\nmodel name\t: procfs-compat\n");
    }
}

static int copy_or_stub_cpuinfo(void) {
    int in_fd;
    int out_fd;
    char buf[16384];
    ssize_t nread;

    in_fd = syscall(SYS_openat, AT_FDCWD, "/proc/cpuinfo", O_RDONLY | O_CLOEXEC, 0);
    if (in_fd >= 0) {
        out_fd = syscall(SYS_openat, AT_FDCWD, g_cpuinfo_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (out_fd < 0) {
            close(in_fd);
            return -1;
        }
        while ((nread = read(in_fd, buf, sizeof(buf))) > 0) {
            if (write(out_fd, buf, (size_t) nread) < 0) {
                int err = errno;
                close(in_fd);
                close(out_fd);
                errno = err;
                return -1;
            }
        }
        close(in_fd);
        close(out_fd);
        return 0;
    }

    build_cpuinfo_fallback(buf, sizeof(buf));
    return write_text_file(g_cpuinfo_path, buf);
}

static int copy_or_stub_mmap_min_addr(void) {
    int in_fd;
    int out_fd;
    char buf[128];
    ssize_t nread;

    in_fd = syscall(SYS_openat, AT_FDCWD, "/proc/sys/vm/mmap_min_addr", O_RDONLY | O_CLOEXEC, 0);
    if (in_fd >= 0) {
        out_fd = syscall(SYS_openat, AT_FDCWD, g_mmap_min_addr_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (out_fd < 0) {
            close(in_fd);
            return -1;
        }
        while ((nread = read(in_fd, buf, sizeof(buf))) > 0) {
            if (write(out_fd, buf, (size_t) nread) < 0) {
                int err = errno;
                close(in_fd);
                close(out_fd);
                errno = err;
                return -1;
            }
        }
        close(in_fd);
        close(out_fd);
        return 0;
    }

    return write_text_file(g_mmap_min_addr_path, "65536\n");
}

static int write_cgroup_stub(void) {
    return write_text_file(g_cgroup_path, "0::/\n");
}

static int write_meminfo_stub(void) {
    struct sysinfo info;
    char buf[1024];

    if (sysinfo(&info) != 0) {
        return write_text_file(
            g_meminfo_path,
            "MemTotal:       1024000 kB\n"
            "MemFree:        1024000 kB\n"
            "MemAvailable:   1024000 kB\n"
            "Buffers:              0 kB\n"
            "Cached:               0 kB\n"
        );
    }

    snprintf(
        buf,
        sizeof(buf),
        "MemTotal:       %8lu kB\n"
        "MemFree:        %8lu kB\n"
        "MemAvailable:   %8lu kB\n"
        "Buffers:        %8lu kB\n"
        "Cached:         %8lu kB\n",
        (unsigned long) (info.totalram * info.mem_unit / 1024),
        (unsigned long) (info.freeram * info.mem_unit / 1024),
        (unsigned long) (info.freeram * info.mem_unit / 1024),
        0UL,
        0UL
    );
    return write_text_file(g_meminfo_path, buf);
}

static int write_comm_stub(void) {
    const char *short_name = program_invocation_short_name;
    char buf[256];

    if (short_name == NULL || short_name[0] == '\0') {
        short_name = "python";
    }
    snprintf(buf, sizeof(buf), "%s\n", short_name);
    return write_text_file(g_comm_path, buf);
}

struct map_writer_ctx {
    FILE *stream;
};

static int map_phdr_cb(struct dl_phdr_info *info, size_t size, void *data) {
    size_t i;
    struct map_writer_ctx *ctx = data;
    const char *name = (info->dlpi_name != NULL && info->dlpi_name[0] != '\0') ? info->dlpi_name : "[exe]";

    (void) size;
    for (i = 0; i < (size_t) info->dlpi_phnum; ++i) {
        const ElfW(Phdr) *phdr = &info->dlpi_phdr[i];
        uintptr_t start;
        uintptr_t end;
        char perms[5];

        if (phdr->p_type != PT_LOAD || phdr->p_memsz == 0) {
            continue;
        }
        start = (uintptr_t) info->dlpi_addr + phdr->p_vaddr;
        end = start + phdr->p_memsz;
        perms[0] = (phdr->p_flags & PF_R) ? 'r' : '-';
        perms[1] = (phdr->p_flags & PF_W) ? 'w' : '-';
        perms[2] = (phdr->p_flags & PF_X) ? 'x' : '-';
        perms[3] = 'p';
        perms[4] = '\0';

        fprintf(
            ctx->stream,
            "%012lx-%012lx %s %08lx 00:00 0 %s\n",
            (unsigned long) start,
            (unsigned long) end,
            perms,
            (unsigned long) phdr->p_offset,
            name
        );
    }
    return 0;
}

static int refresh_maps_file(void) {
    int fd;
    FILE *stream;
    struct map_writer_ctx ctx;
    uintptr_t stack_end;
    uintptr_t stack_start;
    void *heap_end;
    char stack_marker;

    if (g_maps_path[0] == '\0') {
        errno = ENOENT;
        return -1;
    }

    fd = syscall(SYS_openat, AT_FDCWD, g_maps_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) {
        return -1;
    }
    stream = fdopen(fd, "w");
    if (stream == NULL) {
        int err = errno;
        close(fd);
        errno = err;
        return -1;
    }

    ctx.stream = stream;
    dl_iterate_phdr(map_phdr_cb, &ctx);

    heap_end = sbrk(0);
    if (heap_end != (void *) -1 && heap_end != NULL) {
        uintptr_t heap_hi = (uintptr_t) heap_end;
        uintptr_t heap_lo = (heap_hi > (1UL << 20)) ? (heap_hi - (1UL << 20)) : 0;
        fprintf(
            stream,
            "%012lx-%012lx rw-p 00000000 00:00 0 [heap]\n",
            (unsigned long) heap_lo,
            (unsigned long) heap_hi
        );
    }

    stack_end = ((uintptr_t) &stack_marker + 0xfffUL) & ~0xfffUL;
    stack_start = (stack_end > (8UL << 20)) ? (stack_end - (8UL << 20)) : 0;
    fprintf(
        stream,
        "%012lx-%012lx rw-p 00000000 00:00 0 [stack]\n",
        (unsigned long) stack_start,
        (unsigned long) stack_end
    );

    if (fflush(stream) != 0) {
        int err = errno;
        fclose(stream);
        errno = err;
        return -1;
    }
    if (fclose(stream) != 0) {
        return -1;
    }
    return 0;
}

static bool is_proc_task_comm(const char *pathname) {
    const char *p;

    if (pathname == NULL) {
        return false;
    }
    if (strncmp(pathname, "/proc/self/task/", 16) != 0) {
        return false;
    }
    p = pathname + 16;
    if (!isdigit((unsigned char) *p)) {
        return false;
    }
    while (isdigit((unsigned char) *p)) {
        ++p;
    }
    return strcmp(p, "/comm") == 0;
}

static const char *redirect_path(const char *pathname, int flags) {
    if (pathname == NULL || g_root_dir[0] == '\0') {
        return pathname;
    }
    if (strcmp(pathname, "/proc/cpuinfo") == 0) {
        return g_cpuinfo_path;
    }
    if (strcmp(pathname, "/proc/sys/vm/mmap_min_addr") == 0) {
        return g_mmap_min_addr_path;
    }
    if (strcmp(pathname, "/proc/self/cgroup") == 0) {
        return g_cgroup_path;
    }
    if (strcmp(pathname, "/proc/meminfo") == 0) {
        return g_meminfo_path;
    }
    if (strcmp(pathname, "/proc/self/maps") == 0) {
        if (refresh_maps_file() != 0) {
            return pathname;
        }
        return g_maps_path;
    }
    if (is_proc_task_comm(pathname)) {
        if ((flags & O_ACCMODE) == O_WRONLY || (flags & O_ACCMODE) == O_RDWR) {
            return "/dev/null";
        }
        return g_comm_path;
    }
    return pathname;
}

static void init_compat_once(void) {
    const char *tmpdir;
    char template_dir[PATH_MAX];

    g_in_hook = 1;
    resolve_symbols();

    tmpdir = getenv("PROCFS_COMPAT_TMPDIR");
    if (tmpdir == NULL || tmpdir[0] == '\0') {
        tmpdir = getenv("TMPDIR");
    }
    if (tmpdir == NULL || tmpdir[0] == '\0') {
        tmpdir = "/tmp";
    }

    snprintf(template_dir, sizeof(template_dir), "%s/procfs-compat-XXXXXX", tmpdir);
    if (mkdtemp(template_dir) == NULL) {
        g_in_hook = 0;
        return;
    }

    snprintf(g_root_dir, sizeof(g_root_dir), "%s", template_dir);
    snprintf(g_self_dir, sizeof(g_self_dir), "%s/self", g_root_dir);
    snprintf(g_task_dir, sizeof(g_task_dir), "%s/self/task", g_root_dir);
    snprintf(g_sys_vm_dir, sizeof(g_sys_vm_dir), "%s/sys/vm", g_root_dir);
    snprintf(g_cpuinfo_path, sizeof(g_cpuinfo_path), "%s/cpuinfo", g_root_dir);
    snprintf(g_maps_path, sizeof(g_maps_path), "%s/self/maps", g_root_dir);
    snprintf(g_mmap_min_addr_path, sizeof(g_mmap_min_addr_path), "%s/sys/vm/mmap_min_addr", g_root_dir);
    snprintf(g_cgroup_path, sizeof(g_cgroup_path), "%s/self/cgroup", g_root_dir);
    snprintf(g_meminfo_path, sizeof(g_meminfo_path), "%s/meminfo", g_root_dir);
    snprintf(g_comm_path, sizeof(g_comm_path), "%s/self/task/comm", g_root_dir);

    if (mkdir_p(g_task_dir, 0755) != 0 || mkdir_p(g_sys_vm_dir, 0755) != 0) {
        g_root_dir[0] = '\0';
        g_in_hook = 0;
        return;
    }

    copy_or_stub_cpuinfo();
    copy_or_stub_mmap_min_addr();
    write_cgroup_stub();
    write_meminfo_stub();
    write_comm_stub();
    refresh_maps_file();
    g_in_hook = 0;
}

static inline void ensure_initialized(void) {
    if (g_in_hook) {
        return;
    }
    pthread_once(&g_once, init_compat_once);
}

static int syscall_openat_passthrough(int dirfd, const char *pathname, int flags, mode_t mode) {
    return (int) syscall(SYS_openat, dirfd, pathname, flags, mode);
}

int open(const char *pathname, int flags, ...) {
    mode_t mode = 0;
    const char *redirected;

    if (flags & O_CREAT) {
        va_list args;
        va_start(args, flags);
        mode = (mode_t) va_arg(args, int);
        va_end(args);
    }
    ensure_initialized();
    if (g_real_open == NULL || g_in_hook) {
        return syscall_openat_passthrough(AT_FDCWD, pathname, flags, mode);
    }
    g_in_hook = 1;
    redirected = redirect_path(pathname, flags);
    g_in_hook = 0;
    return (flags & O_CREAT) ? g_real_open(redirected, flags, mode) : g_real_open(redirected, flags);
}

int open64(const char *pathname, int flags, ...) {
    mode_t mode = 0;
    const char *redirected;

    if (flags & O_CREAT) {
        va_list args;
        va_start(args, flags);
        mode = (mode_t) va_arg(args, int);
        va_end(args);
    }
    ensure_initialized();
    if (g_real_open64 == NULL || g_in_hook) {
        return syscall_openat_passthrough(AT_FDCWD, pathname, flags, mode);
    }
    g_in_hook = 1;
    redirected = redirect_path(pathname, flags);
    g_in_hook = 0;
    return (flags & O_CREAT) ? g_real_open64(redirected, flags, mode) : g_real_open64(redirected, flags);
}

int openat(int dirfd, const char *pathname, int flags, ...) {
    mode_t mode = 0;
    const char *redirected;

    if (flags & O_CREAT) {
        va_list args;
        va_start(args, flags);
        mode = (mode_t) va_arg(args, int);
        va_end(args);
    }
    ensure_initialized();
    if (g_real_openat == NULL || g_in_hook) {
        return syscall_openat_passthrough(dirfd, pathname, flags, mode);
    }
    g_in_hook = 1;
    redirected = redirect_path(pathname, flags);
    g_in_hook = 0;
    return (flags & O_CREAT) ? g_real_openat(dirfd, redirected, flags, mode) : g_real_openat(dirfd, redirected, flags);
}

int openat64(int dirfd, const char *pathname, int flags, ...) {
    mode_t mode = 0;
    const char *redirected;

    if (flags & O_CREAT) {
        va_list args;
        va_start(args, flags);
        mode = (mode_t) va_arg(args, int);
        va_end(args);
    }
    ensure_initialized();
    if (g_real_openat64 == NULL || g_in_hook) {
        return syscall_openat_passthrough(dirfd, pathname, flags, mode);
    }
    g_in_hook = 1;
    redirected = redirect_path(pathname, flags);
    g_in_hook = 0;
    return (flags & O_CREAT) ? g_real_openat64(dirfd, redirected, flags, mode) : g_real_openat64(dirfd, redirected, flags);
}

FILE *fopen(const char *pathname, const char *mode) {
    const char *redirected;
    int flags = O_RDONLY;

    ensure_initialized();
    if (g_real_fopen == NULL || g_in_hook) {
        return NULL;
    }
    if (mode != NULL && strchr(mode, '+') != NULL) {
        flags = O_RDWR;
    } else if (mode != NULL && (strchr(mode, 'w') != NULL || strchr(mode, 'a') != NULL)) {
        flags = O_WRONLY;
    }
    g_in_hook = 1;
    redirected = redirect_path(pathname, flags);
    g_in_hook = 0;
    return g_real_fopen(redirected, mode);
}

FILE *fopen64(const char *pathname, const char *mode) {
    const char *redirected;
    int flags = O_RDONLY;

    ensure_initialized();
    if (g_real_fopen64 == NULL || g_in_hook) {
        return NULL;
    }
    if (mode != NULL && strchr(mode, '+') != NULL) {
        flags = O_RDWR;
    } else if (mode != NULL && (strchr(mode, 'w') != NULL || strchr(mode, 'a') != NULL)) {
        flags = O_WRONLY;
    }
    g_in_hook = 1;
    redirected = redirect_path(pathname, flags);
    g_in_hook = 0;
    return g_real_fopen64(redirected, mode);
}

int access(const char *pathname, int mode) {
    const char *redirected;

    ensure_initialized();
    if (g_real_access == NULL || g_in_hook) {
        return faccessat(AT_FDCWD, pathname, mode, 0);
    }
    g_in_hook = 1;
    redirected = redirect_path(pathname, O_RDONLY);
    g_in_hook = 0;
    return g_real_access(redirected, mode);
}

int faccessat(int dirfd, const char *pathname, int mode, int flags) {
    const char *redirected;

    ensure_initialized();
    if (g_real_faccessat == NULL || g_in_hook) {
        return (int) syscall(SYS_faccessat, dirfd, pathname, mode, flags);
    }
    g_in_hook = 1;
    redirected = redirect_path(pathname, O_RDONLY);
    g_in_hook = 0;
    return g_real_faccessat(dirfd, redirected, mode, flags);
}

int stat(const char *pathname, struct stat *buf) {
    const char *redirected;

    ensure_initialized();
    if (g_real_stat == NULL || g_in_hook) {
        return (int) syscall(SYS_newfstatat, AT_FDCWD, pathname, buf, 0);
    }
    g_in_hook = 1;
    redirected = redirect_path(pathname, O_RDONLY);
    g_in_hook = 0;
    return g_real_stat(redirected, buf);
}

int lstat(const char *pathname, struct stat *buf) {
    const char *redirected;

    ensure_initialized();
    if (g_real_lstat == NULL || g_in_hook) {
        return (int) syscall(SYS_newfstatat, AT_FDCWD, pathname, buf, AT_SYMLINK_NOFOLLOW);
    }
    g_in_hook = 1;
    redirected = redirect_path(pathname, O_RDONLY);
    g_in_hook = 0;
    return g_real_lstat(redirected, buf);
}

int stat64(const char *pathname, struct stat64 *buf) {
    const char *redirected;

    ensure_initialized();
    if (g_real_stat64 == NULL || g_in_hook) {
        return -1;
    }
    g_in_hook = 1;
    redirected = redirect_path(pathname, O_RDONLY);
    g_in_hook = 0;
    return g_real_stat64(redirected, buf);
}

int lstat64(const char *pathname, struct stat64 *buf) {
    const char *redirected;

    ensure_initialized();
    if (g_real_lstat64 == NULL || g_in_hook) {
        return -1;
    }
    g_in_hook = 1;
    redirected = redirect_path(pathname, O_RDONLY);
    g_in_hook = 0;
    return g_real_lstat64(redirected, buf);
}

int __xstat(int ver, const char *pathname, struct stat *buf) {
    const char *redirected;

    ensure_initialized();
    if (g_real___xstat == NULL || g_in_hook) {
        return -1;
    }
    g_in_hook = 1;
    redirected = redirect_path(pathname, O_RDONLY);
    g_in_hook = 0;
    return g_real___xstat(ver, redirected, buf);
}

int __lxstat(int ver, const char *pathname, struct stat *buf) {
    const char *redirected;

    ensure_initialized();
    if (g_real___lxstat == NULL || g_in_hook) {
        return -1;
    }
    g_in_hook = 1;
    redirected = redirect_path(pathname, O_RDONLY);
    g_in_hook = 0;
    return g_real___lxstat(ver, redirected, buf);
}

int __xstat64(int ver, const char *pathname, struct stat64 *buf) {
    const char *redirected;

    ensure_initialized();
    if (g_real___xstat64 == NULL || g_in_hook) {
        return -1;
    }
    g_in_hook = 1;
    redirected = redirect_path(pathname, O_RDONLY);
    g_in_hook = 0;
    return g_real___xstat64(ver, redirected, buf);
}

int __lxstat64(int ver, const char *pathname, struct stat64 *buf) {
    const char *redirected;

    ensure_initialized();
    if (g_real___lxstat64 == NULL || g_in_hook) {
        return -1;
    }
    g_in_hook = 1;
    redirected = redirect_path(pathname, O_RDONLY);
    g_in_hook = 0;
    return g_real___lxstat64(ver, redirected, buf);
}

int newfstatat(int dirfd, const char *pathname, struct stat *buf, int flags) {
    const char *redirected;

    ensure_initialized();
    if (g_real_newfstatat == NULL || g_in_hook) {
        return (int) syscall(SYS_newfstatat, dirfd, pathname, buf, flags);
    }
    g_in_hook = 1;
    redirected = redirect_path(pathname, O_RDONLY);
    g_in_hook = 0;
    return g_real_newfstatat(dirfd, redirected, buf, flags);
}

int statx(int dirfd, const char *pathname, int flags, unsigned int mask, struct statx *buf) {
    const char *redirected;

    ensure_initialized();
    if (g_real_statx == NULL || g_in_hook) {
        return -1;
    }
    g_in_hook = 1;
    redirected = redirect_path(pathname, O_RDONLY);
    g_in_hook = 0;
    return g_real_statx(dirfd, redirected, flags, mask, buf);
}
