import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 数据预处理
def load_data():
    data = [
        (
        "3849   bash               3   0 /usr/share/locale/locale.alias\n3849   bash              -1   2 /usr/share/locale/en_US.UTF-8/LC_MESSAGES/libc.mo\n3849   bash              -1   2 /usr/share/locale/en_US.utf8/LC_MESSAGES/libc.mo",
        1),
        (
        "3933   docker             3   0 /etc/ld.so.cache\n3933   docker             3   0 /lib/x86_64-linux-gnu/libresolv.so.2\n3933   docker             3   0 /lib/x86_64-linux-gnu/libpthread.so.0",
        1),
        (
        "4056   clang             -1   2 /usr/bin/../lib/gcc/x86_64-amazon-linux\n4056   clang             -1   2 /usr/bin/../lib/gcc/x86_64-pc-linux-gnu",
        1),
        (
        "2359   cat                3   0 /etc/ld.so.cache\n2359   cat                3   0 /lib/x86_64-linux-gnu/libc.so.6\n2359   cat                3   0 /usr/lib/locale/locale-archive",
        1),
        (
        "2398   dd                 4   0 /etc/ld.so.cache\n2398   dd                 4   0 /lib/x86_64-linux-gnu/libc.so.6\n2398   dd                 4   0 /usr/lib/locale/locale-archive",
        1),
        ("2281   pool-org.gnome.   -1  13 /home/ubuntu/Desktop/1.txt", 1),
        ("18372  run       -1   6 /dev/tty\n18373  run       -1   6 /dev/tty", 1),
        (
        "18384  df        -1   2 /usr/share/locale/en_US.UTF-8/LC_MESSAGES/coreutils.mo\n18384  df        -1   2 /usr/share/locale/en_US.utf8/LC_MESSAGES/coreutils.mo",
        1),
        (
        "747    irqbalance         6   0 /proc/interrupts\n747    irqbalance         6   0 /proc/stat\n747    irqbalance         6   0 /proc/irq/57/smp_affinity",
        0),
        (
        "715    vmtoolsd           9   0 /etc/mtab\n715    vmtoolsd          11   0 /proc/devices\n715    vmtoolsd           9   0 /sys/class/block/sda5/../device/../../../class",
        0),
        (
        "2325   gnome-terminal-   22   0 /usr/share/icons/Yaru/scalable/actions/open-menu-symbolic.svg\n2325   gnome-terminal-   22   0 /usr/share/icons/Yaru/scalable/actions/edit-find-symbolic.svg",
        0),
        (
        "691   getconf            3   0 /etc/ld.so.cache\n3691   getconf            3   0 /lib/x86_64-linux-gnu/libc.so.6\n3684   chaosd             6   0 /etc/nsswitch.conf",
        0),
        (
        "3692   bash               3   0 /etc/ld.so.cache\n3692   bash               3   0 /lib/x86_64-linux-gnu/libtinfo.so.6\n3692   bash               3   0 /lib/x86_64-linux-gnu/libc.so.6\n3692   bash               3   0 /dev/tty",
        0),
        (
        "3811   sudo               5   0 /etc/pam.d/common-session-noninteractive\n3811   sudo               6   0 /lib/x86_64-linux-gnu/security/pam_umask.so\n3811   sudo               4   0 /etc/pam.d/other",
        0),
        (
        "3811   sudo               4   0 /etc/passwd\n3811   sudo               4   0 /etc/passwd\n3811   sudo               4   0 /etc/login.defs\n3811   sudo               4   0 /etc/passwd",
        0),
        (
        "3812   su                 3   0 /proc/self/loginuid\n3812   su                 3   0 /etc/passwd\n3812   su                 3   0 /etc/pam.d/su\n3812   su                 4   0 /lib/x86_64-linux-gnu/security/pam_rootok.so\n3812   su                 4   0 /etc/ld.so.cache",
        0),
        (
        "3814   groups             3   0 /etc/ld.so.cache\n3814   groups             3   0 /lib/x86_64-linux-gnu/libc.so.6\n3814   groups             3   0 /usr/lib/locale/locale-archive\n3814   groups             3   0 /etc/nsswitch.conf\n3814   groups             3   0 /etc/ld.so.cache",
        0),
        (
        "3818   dirname            3   0 /etc/ld.so.cache\n3818   dirname            3   0 /lib/x86_64-linux-gnu/libc.so.6\n3818   dirname            3   0 /usr/lib/locale/locale-archive\n3819   dircolors          3   0 /etc/ld.so.cache",
        0),
        (
        "3820   vim                3   0 /etc/ld.so.cache\n3820   vim                3   0 /lib/x86_64-linux-gnu/libm.so.6\n3820   vim                3   0 /lib/x86_64-linux-gnu/libtinfo.so.6\n3820   vim                3   0 /lib/x86_64-linux-gnu/libselinux.so.1\n3820   vim                3   0 /lib/x86_64-linux-gnu/libcanberra.so.0\n3820   vim                3   0 /lib/x86_64-linux-gnu/libacl.so.1\n3820   vim                3   0 /lib/x86_64-linux-gnu/libgpm.so.2\n3820   vim                3   0 /lib/x86_64-linux-gnu/libdl.so.2\n3820   vim                3   0 /lib/x86_64-linux-gnu/libpython3.8.so.1.0\n3820   vim                3   0 /lib/x86_64-linux-gnu/libpthread.so.0\n3820   vim                3   0 /lib/x86_64-linux-gnu/libc.so.6\n3820   vim                3   0 /lib/x86_64-linux-gnu/libpcre2-8.so.0\n3820   vim                3   0 /lib/x86_64-linux-gnu/libvorbisfile.so.3\n3820   vim                3   0 /lib/x86_64-linux-gnu/libtdb.so.1\n3820   vim                3   0 /lib/x86_64-linux-gnu/libltdl.so.7\n3820   vim                3   0 /lib/x86_64-linux-gnu/libexpat.so.1",
        0),
        (
        "1      systemd           -1   2 /sys/fs/cgroup/memory/system.slice/fprintd.service/memory.events\n1      systemd           89   0 /sys/fs/cgroup/unified/system.slice/fprintd.service/cgroup.procs\n1      systemd           89   0 /sys/fs/cgroup/unified/system.slice/fprintd.service",
        0),
        (
        "1684   pulseaudio        47   0 /dev/snd/controlC0\n1684   alsa-sink-ES137   47   0 /dev/snd/controlC0\n1684   alsa-sink-ES137   47   0 /dev/snd/controlC0\n1684   alsa-sink-ES137   47   0 /dev/snd/controlC0\n1684   alsa-sink-ES137   47   0 /dev/snd/controlC0\n1684   alsa-sink-ES137   50   0 /dev/snd/pcmC0D0p\n1684   alsa-sink-ES137   47   0 /dev/snd/timer",
        0),
        (
        "3832   tracker-store     13   0 /usr/share/tracker/ontologies/nepomuk/30-nie.ontology\n3832   tracker-store     13   0 /usr/share/tracker/ontologies/nepomuk/32-nco.ontology\n3832   tracker-store     13   0 /usr/share/tracker/ontologies/nepomuk/33-nfo.ontology\n3832   tracker-store     13   0 /usr/share/tracker/ontologies/nepomuk/34-nmo.ontology\n3832   tracker-store     13   0 /usr/share/tracker/ontologies/nepomuk/35-ncal.ontology\n3832   tracker-store     13   0 /usr/share/tracker/ontologies/nepomuk/36-scal.ontology\n3832   tracker-store     13   0 /usr/share/tracker/ontologies/nepomuk/37-nid3.ontology\n3832   tracker-store     13   0 /usr/share/tracker/ontologies/nepomuk/38-nmm.ontology\n3832   tracker-store     13   0 /usr/share/tracker/ontologies/nepomuk/39-mto.ontology\n3832   tracker-store     13   0 /usr/share/tracker/ontologies/nepomuk/40-mlo.ontology\n3832   tracker-store     13   0 /usr/share/tracker/ontologies/nepomuk/41-mfo.ontology\n3832   tracker-store     13   0 /usr/share/tracker/ontologies/nepomuk/89-mtp.ontology\n3832   tracker-store     13   0 /usr/share/tracker/ontologies/nepomuk/90-tracker.ontology\n3832   tracker-store     13   0 /usr/share/tracker/ontologies/nepomuk/91-maemo.ontology\n3832   tracker-store     13   0 /usr/share/tracker/ontologies/nepomuk/92-slo.ontology\n3832   tracker-store     13   0 /usr/share/tracker/ontologies/nepomuk/93-libosinfo.ontology\n3832   tracker-store     13   0 /home/ubuntu/.local/share/tracker/data/tracker-store.journal\n3832   tracker-store     14   0 /home/ubuntu/.cache/tracker/db-locale.txt\n3832   tracker-store     14   0 /home/ubuntu/.cache/tracker/parser-version.txt\n3832   tracker-store     12   0 /home/ubuntu/.cache/tracker/meta.db\n3832   tracker-store     14   0 /home/ubuntu/.cache/tracker/meta.db-wal",
        0),
        (
        "3866   docker             3   0 /etc/ld.so.cache\n3866   docker             3   0 /lib/x86_64-linux-gnu/libresolv.so.2\n3866   docker             3   0 /lib/x86_64-linux-gnu/libpthread.so.0\n3866   docker             3   0 /lib/x86_64-linux-gnu/libdl.so.2\n3866   docker             3   0 /lib/x86_64-linux-gnu/libc.so.6\n3866   docker             3   0 /sys/kernel/mm/transparent_hugepage/hpage_pmd_size\n3866   docker             3   0 /usr/bin/docker\n1157   dockerd           28   0 /var/lib/docker/containers/781d62ccc0989557d438af4d294e6b355fb56ecf1ba3e7a0c4fbb50cd4901962/.tmp-config.v2.json2397377541\n1157   dockerd           29   0 /var/lib/docker/containers/781d62ccc0989557d438af4d294e6b355fb56ecf1ba3e7a0c4fbb50cd4901962/.tmp-hostconfig.json821890836\n2872   containerd-shim   23   0 /dev/null",
        0),
        (
        "3873   runc              -1   2 tls/haswell/libc.so.6\ntls/x86_64/libc.so.6\n3873   runc              -1   2 tls/libc.so.6\n3873   runc              -1   2 haswell/x86_64/libc.so.6\n3873   runc              -1   2 haswell/libc.so.6\n3873   runc              -1   2 x86_64/libc.so.6\n3873   runc              -1   2 libc.so.6\n3873   runc               3   0 /lib/x86_64-linux-gnu/libc.so.6\n3873   runc               3   0 /sys/kernel/mm/transparent_hugepage/hpage_pmd_size\n3873   runc               3   0 /usr/bin/runc\n3873   runc               3   0 /proc/sys/kernel/cap_last_cap",
        0),
        (
        "3910   modprobe           0   0 /etc/ld.so.cache\n3910   modprobe           0   0 /lib/x86_64-linux-gnu/libzstd.so.1\n3910   modprobe           0   0 /lib/x86_64-linux-gnu/liblzma.so.5\n3910   modprobe           0   0 /lib/x86_64-linux-gnu/libcrypto.so.1.1\n3910   modprobe           0   0 /lib/x86_64-linux-gnu/libc.so.6\n3910   modprobe           0   0 /lib/x86_64-linux-gnu/libpthread.so.0\n3910   modprobe           0   0 /lib/x86_64-linux-gnu/libdl.so.2\n3910   modprobe           0   0 /etc/modprobe.d\n3910   modprobe           0   0 /lib/modprobe.d\n3910   modprobe           0   0 /lib/modprobe.d/aliases.conf\n3910   modprobe           0   0 /etc/modprobe.d/alsa-base.conf\n3910   modprobe           0   0 /etc/modprobe.d/amd64-microcode-blacklist.conf\n3910   modprobe           0   0 /etc/modprobe.d/blacklist-ath_pci.conf\n3910   modprobe           0   0 /etc/modprobe.d/blacklist-firewire.conf\n3910   modprobe           0   0 /etc/modprobe.d/blacklist-framebuffer.conf\n3910   modprobe           0   0 /etc/modprobe.d/blacklist-modem.conf\n3910   modprobe           0   0 /etc/modprobe.d/blacklist-oss.conf",
        0),
        (
        "1      systemd           96   0 /\n1      systemd           97   0 sys\n1      systemd           96   0 dev\n1      systemd           97   0 block\n1      systemd           96   0 7:4\n1      systemd           96   0 ..\n1      systemd           97   0 ..\n1      systemd           96   0 devices\n1      systemd           97   0 virtual\n1      systemd           96   0 block\n1      systemd           97   0 loop4\n1      systemd           96   0 /sys/devices/virtual/block/loop4/uevent\n1      systemd           96   0 /run/udev/data/b7:4",
        0),
        (
        "3985   git-remote-http    7   0 /etc/ssl/certs/8d89cda1.0\n3985   git-remote-http    7   0 /etc/ssl/certs/4b718d9b.0\n3985   git-remote-http    7   0 /etc/ssl/certs/064e0aa9.0\n3985   git-remote-http    7   0 /etc/ssl/certs/eed8c118.0\n3985   git-remote-http    7   0 /etc/ssl/certs/Hellenic_Academic_and_Research_Institutions_ECC_RootCA_2015.pem\n3985   git-remote-http    7   0 /etc/ssl/certs/3fb36b73.0\n3985   git-remote-http    7   0 /etc/ssl/certs/QuoVadis_Root_CA_2.pem\n3985   git-remote-http    7   0 /etc/ssl/certs/4f316efb.0\n3985   git-remote-http    7   0 /etc/ssl/certs/IdenTrust_Commercial_Root_CA_1.pem\n3985   git-remote-http    7   0 /etc/ssl/certs/SecureSign_RootCA11.pem\n3985   git-remote-http    7   0 /etc/ssl/certs/6d41d539.0\n3985   git-remote-http    7   0 /etc/ssl/certs/48bec511.0\n3985   git-remote-http    7   0 /etc/ssl/certs/0c31d5ce\n3985   git-remote-http    7   0 /etc/ssl/certs/SSL.com_Root_Certification_Authority_RSA.pem\n3985   git-remote-http    7   0 /etc/ssl/certs/Certum_Trusted_Network_CA_2.pem\n3985   git-remote-http    7   0 /etc/ssl/certs/Amazon_Root_CA_2.pem\n3985   git-remote-http    7   0 /etc/ssl/certs/Izenpe.com.pem\n3985   git-remote-http    7   0 /etc/ssl/certs/773e07ad.0",
        0),
        (
        "3984   git               -1   2 /usr/share/locale/en.utf8/LC_MESSAGES/git.mo\n3984   git               -1   2 /usr/share/locale/en/LC_MESSAGES/git.mo\n3984   git               -1   2 /usr/share/locale-langpack/en_US.UTF-8/LC_MESSAGES/git.mo\n3984   git               -1   2 /usr/share/locale-langpack/en_US.utf8/LC_MESSAGES/git.mo\n3984   git               -1   2 /usr/share/locale-langpack/en_US/LC_MESSAGES/git.mo\n3984   git               -1   2 /usr/share/locale-langpack/en.UTF-8/LC_MESSAGES/git.mo\n3984   git               -1   2 /usr/share/locale-langpack/en.utf8/LC_MESSAGES/git.mo\n3984   git               -1   2 /usr/share/locale-langpack/en/LC_MESSAGES/git.mo\n3984   git               -1   2 /home/ubuntu/Desktop/nydus/.git/config\n3984   git                3   0 /usr/share/git-core/templates/\n3984   git               -1   2 /usr/share/git-core/templates/config\n3984   git                4   0 /usr/share/git-core/templates/hooks\n3984   git                5   0 /usr/share/git-core/templates/hooks/update.sample\n3984   git                6   0 /home/ubuntu/Desktop/nydus/.git/hooks/update.sample\n3984   git                5   0 /usr/share/git-core/templates/hooks/pre-rebase.sample\n3984   git                6   0 /home/ubuntu/Desktop/nydus/.git/hooks/pre-rebase.sample\n3984   git                5   0 /usr/share/git-core/templates/hooks/pre-applypatch.sample\n3984   git                6   0 /home/ubuntu/Desktop/nydus/.git/hooks/pre-applypatch.sample\n3984   git                5   0 /usr/share/git-core/templates/hooks/fsmonitor-watchman.sample",
        0),
        (
        "2281   pool-org.gnome.   36   0 /home/ubuntu/Documents/project (copy)/bpftools/biosnoop/.output/bpf/bpf_tracing.h\n2281   pool-org.gnome.   35   0 /home/ubuntu/Documents/project/bpftools/biosnoop/.output/bpf/bpf_endian.h\n2281   pool-org.gnome.   36   0 /home/ubuntu/Documents/project (copy)/bpftools/biosnoop/.output/bpf/bpf_endian.h\n2281   pool-org.gnome.   35   0 /home/ubuntu/Documents/project/bpftools/biosnoop/.output/bpf/bpf_core_read.h\n2281   pool-org.gnome.   36   0 /home/ubuntu/Documents/project (copy)/bpftools/biosnoop/.output/bpf/bpf_core_read.h\n2281   pool-org.gnome.   35   0 /home/ubuntu/Documents/project/bpftools/biosnoop/.output/bpf/skel_internal.h\n2281   pool-org.gnome.   36   0 /home/ubuntu/Documents/project (copy)/bpftools/biosnoop/.output/bpf/skel_internal.h\n2281   pool-org.gnome.   35   0 /home/ubuntu/Documents/project/bpftools/biosnoop/.output/bpf/libbpf_version.h\n2281   pool-org.gnome.   36   0 /home/ubuntu/Documents/project (copy)/bpftools/biosnoop/.output/bpf/libbpf_version.h\n2281   pool-org.gnome.   35   0 /home/ubuntu/Documents/project/bpftools/biosnoop/.output/bpf/usdt.bpf.h\n2281   pool-org.gnome.   36   0 /home/ubuntu/Documents/project (copy)/bpftools/biosnoop/.output/bpf/usdt.bpf.h\n2281   pool-org.gnome.   34   0 /home/ubuntu/Documents/project/bpftools/biosnoop/.output/pkgconfig\n2281   pool-org.gnome.   34   0 /home/ubuntu/Documents/project/bpftools/biosnoop/.output/pkgconfig",
        0),
        (
        "4023   single            17   0 /home/ubuntu/Documents/project (copy)/bpftools/biolatency/biolatency_tracker.h\n2213   gvfsd-metadata    11   0 /\n2213   gvfsd-metadata    12   0 sys\n2213   gvfsd-metadata    11   0 dev\n2213   gvfsd-metadata    12   0 block\n2213   gvfsd-metadata    11   0 8:5\n2213   gvfsd-metadata    11   0 ..\n2213   gvfsd-metadata    12   0 ..\n2213   gvfsd-metadata    11   0 devices\n2213   gvfsd-metadata    12   0 pci0000:00\n2213   gvfsd-metadata    11   0 0000:00:10.0\n2213   gvfsd-metadata    12   0 host32\n2213   gvfsd-metadata    11   0 target32:0:0\n2213   gvfsd-metadata    12   0 32:0:0:0\n2213   gvfsd-metadata    11   0 block",
        0),
        (
        "2281   nautilus          30   0 /usr/share/icons/Yaru/scalable/actions/window-minimize-symbolic.svg\n2281   nautilus          30   0 /usr/share/icons/Yaru/scalable/actions/window-maximize-symbolic.svg\n2281   nautilus          30   0 /usr/share/icons/Yaru/scalable/actions/document-open-recent-symbolic.svg\n2281   nautilus          30   0 /usr/share/icons/Yaru/scalable/status/starred-symbolic.svg\n2281   nautilus          30   0 /usr/share/icons/Yaru/scalable/places/user-home-symbolic.svg\n2281   nautilus          30   0 /usr/share/icons/Yaru/scalable/places/user-desktop-symbolic.svg\n2281   nautilus          30   0 /usr/share/icons/Yaru/scalable/places/folder-documents-symbolic.svg\n2281   nautilus          30   0 /usr/share/icons/Yaru/scalable/places/folder-download-symbolic.svg\n2281   nautilus          30   0 /usr/share/icons/Yaru/scalable/places/folder-music-symbolic.svg\n2281   nautilus          30   0 /usr/share/icons/Yaru/scalable/places/folder-pictures-symbolic.svg\n2281   nautilus          30   0 /usr/share/icons/Yaru/scalable/places/folder-videos-symbolic.svg\n2281   nautilus          30   0 /usr/share/icons/Yaru/scalable/status/user-trash-full-symbolic.svg\n2281   nautilus          30   0 /usr/share/icons/Yaru/scalable/devices/media-floppy-symbolic.svg\n2281   nautilus          30   0 /usr/share/icons/Yaru/scalable/devices/media-optical-symbolic.svg",
        0),
        (
        "4706   tracker-store     -1   2 /usr/share/ubuntu/glib-2.0/schemas/gschemas.compiled\n4706   tracker-store     -1   2 /home/ubuntu/.local/share/glib-2.0/schemas/gschemas.compiled\n4706   tracker-store      4   0 /usr/lib/x86_64-linux-gnu/gio/modules\n4706   tracker-store      5   0 /usr/lib/x86_64-linux-gnu/gio/modules/giomodule.cache\n4706   tracker-store      4   0 /usr/lib/x86_64-linux-gnu/gio/modules/libdconfsettings.so\n4706   tracker-store     -1   2 /run/dconf/user/1000\n4706   tracker-store     -1   2 /run/user/1000/dconf/profile\n4706   tracker-store     -1   2 /etc/dconf/profile/user\n4706   tracker-store     -1   2 /usr/share/ubuntu/dconf/profile/user\n4706   tracker-store     -1   2 /usr/local/share/dconf/profile/user\n4706   tracker-store     -1   2 /usr/share/dconf/profile/user\n4706   tracker-store     -1   2 /var/lib/snapd/desktop/dconf/profile/user\n4706   tracker-store      4   0 /run/user/1000/dconf/user\n4706   tracker-store      4   0 /home/ubuntu/.config/dconf/user",
        0),
        (
        "4706   (er-store)        -1  13 /sys/fs/cgroup/memory/user.slice/user-1000.slice/user@1000.service/cgroup.procs\n4706   (er-store)        -1  13 /sys/fs/cgroup/memory/user.slice/user-1000.slice/cgroup.procs\n4706   (er-store)        -1  13 /sys/fs/cgroup/memory/user.slice/cgroup.procs\n4706   (er-store)        -1  13 /sys/fs/cgroup/memory/cgroup.procs\n4706   (er-store)        -1   2 /sys/fs/cgroup/devices/user.slice/user-1000.slice/user@1000.service/tracker-store.service/cgroup.procs\n4706   (er-store)        -1   2 /sys/fs/cgroup/devices/user.slice/user-1000.slice/cgroup.procs\n4706   (er-store)        -1   2 /sys/fs/cgroup/devices/user.slice/cgroup.procs\n4706   (er-store)        -1  13 /sys/fs/cgroup/devices/user.slice/cgroup.procs\n4706   (er-store)        -1  13 /sys/fs/cgroup/devices/cgroup.procs\n4706   (er-store)        -1   2 /sys/fs/cgroup/pids/user.slice/user-1000.slice/user@1000.service/tracker-store.service/cgroup.procs\n4706   (er-store)        -1  13 /sys/fs/cgroup/pids/user.slice/user-1000.slice/user@1000.service/cgroup.procs\n4706   (er-store)        -1  13 /sys/fs/cgroup/pids/user.slice/user-1000.slice/cgroup.procs\n4706   (er-store)        -1  13 /sys/fs/cgroup/pids/user.slice/cgroup.procs\n4706   (er-store)        -1  13 /sys/fs/cgroup/pids/cgroup.procs\n4706   (er-store)         3   0 /dev/null",
        0),
        (
        "754    polkitd           11   0 /proc/1811/stat\n754    polkitd           11   0 /proc/1811/cgroup\n754    polkitd           11   0 /proc/1/cgroup\n754    polkitd           11   0 /proc/1811/cgroup\n754    polkitd           11   0 /proc/1/cgroup\n754    polkitd           11   0 /run/systemd/users/1000\n754    polkitd           11   0 /run/systemd/sessions/2\n754    polkitd           11   0 /run/systemd/sessions/2\n754    polkitd           11   0 /run/systemd/users/1000\n754    polkitd           11   0 /etc/passwd\n754    polkitd           11   0 /etc/group",
        0),
        (
        "4832   ld                -1   2 /usr/share/locale/en_US.UTF-8/LC_MESSAGES/bfd.mo\n4832   ld                -1   2 /usr/share/locale/en_US.utf8/LC_MESSAGES/bfd.mo\n4832   ld                -1   2 /usr/share/locale/en_US/LC_MESSAGES/bfd.mo\n4832   ld                -1   2 /usr/share/locale/en.UTF-8/LC_MESSAGES/bfd.mo\n4832   ld                -1   2 /usr/share/locale/en.utf8/LC_MESSAGES/bfd.mo\n4832   ld                -1   2 /usr/share/locale/en/LC_MESSAGES/bfd.mo\n4832   ld                -1   2 /usr/share/locale-langpack/en_US.UTF-8/LC_MESSAGES/bfd.mo\n4832   ld                -1   2 /usr/share/locale-langpack/en_US.utf8/LC_MESSAGES/bfd.mo\n4832   ld                -1   2 /usr/share/locale-langpack/en_US/LC_MESSAGES/bfd.mo\n4832   ld                -1   2 /usr/share/locale-langpack/en.UTF-8/LC_MESSAGES/bfd.mo\n4832   ld                -1   2 /usr/share/locale-langpack/en.utf8/LC_MESSAGES/bfd.mo\n4832   ld                -1   2 /usr/share/locale-langpack/en/LC_MESSAGES/bfd.mo",
        0),
        (
        "4832   ld                -1   2 /usr/lib/gcc/x86_64-linux-gnu/10/libgcc.so\n4832   ld                28   0 /usr/lib/gcc/x86_64-linux-gnu/10/libgcc.a\n4832   ld                29   0 /usr/lib/gcc/x86_64-linux-gnu/10/libgcc_s.so\n4832   ld                30   0 /usr/lib/gcc/x86_64-linux-gnu/10/libgcc_s.so\n4832   ld                31   0 /usr/lib/gcc/x86_64-linux-gnu/10/libgcc_s.so\n4832   ld                29   0 /usr/lib/gcc/x86_64-linux-gnu/10/libgcc_s.so\n4832   ld                -1   2 libgcc_s.so.1\n4832   ld                -1   2 /usr/lib/gcc/x86_64-linux-gnu/10/libgcc_s.so.1\n4832   ld                29   0 /usr/lib/gcc/x86_64-linux-gnu/10/../../../x86_64-linux-gnu/libgcc_s.so.1\n4832   ld                31   0 /usr/lib/gcc/x86_64-linux-gnu/10/../../../x86_64-linux-gnu/libgcc_s.so.1\n4832   ld                -1   2 /usr/lib/gcc/x86_64-linux-gnu/10/libgcc.so\n4832   ld                31   0 /usr/lib/gcc/x86_64-linux-gnu/10/libgcc.a\n4832   ld                32   0 /usr/lib/gcc/x86_64-linux-gnu/10/crtendS.o\n4832   ld                33   0 /usr/lib/gcc/x86_64-linux-gnu/10/crtendS.o",
        0),
        (
        "4827   cc1                6   0 .output/memleak.skel.h\n4827   cc1                7   0 blazesym.h\n4827   cc1                7   0 memleak.c\n4828   as                 3   0 /etc/ld.so.cache\n4828   as                 3   0 /lib/x86_64-linux-gnu/libopcodes-2.34-system.so\n4828   as                 3   0 /lib/x86_64-linux-gnu/libbfd-2.34-system.so\n4828   as                 3   0 /lib/x86_64-linux-gnu/libz.so.1\n4828   as                 3   0 /lib/x86_64-linux-gnu/libc.so.6\n4828   as                 3   0 /lib/x86_64-linux-gnu/libdl.so.2\n4828   as                 3   0 /usr/lib/locale/locale-archive\n4828   as                 3   0 .output/memleak.o\n4828   as                 4   0 /tmp/cciAiOT8.s\n4828   as                 4   0 /usr/share/locale/locale.alias\n4828   as                -1   2 /usr/share/locale/en_US.UTF-8/LC_MESSAGES/gas.mo\n4828   as                -1   2 /usr/share/locale/en_US.utf8/LC_MESSAGES/gas.mo\n4828   as                -1   2 /usr/share/locale/en_US/LC_MESSAGES/gas.mo\n4828   as                -1   2 /usr/share/locale/en.UTF-8/LC_MESSAGES/gas.mo\n4828   as                -1   2 /usr/share/locale/en.utf8/LC_MESSAGES/gas.mo\n4828   as                -1   2 /usr/share/locale/en/LC_MESSAGES/gas.mo",
        0),
        (
        "4786   make               3   0 /etc/ld.so.cache\n4786   make               3   0 /lib/x86_64-linux-gnu/libdl.so.2\n4786   make               3   0 /lib/x86_64-linux-gnu/libc.so.6\n4786   make               3   0 /usr/lib/locale/locale-archive\n4786   make               3   0 .\n4786   make               3   0 Makefile\n4787   sh                 3   0 /etc/ld.so.cache\n4787   sh                 3   0 /lib/x86_64-linux-gnu/libc.so.6\n4788   uname              3   0 /etc/ld.so.cache\n4788   uname              3   0 /lib/x86_64-linux-gnu/libc.so.6\n4795   sed                3   0 /etc/ld.so.cache\n4795   sed                3   0 /lib/x86_64-linux-gnu/libacl.so.1\n4795   sed                3   0 /lib/x86_64-linux-gnu/libselinux.so.1\n4795   sed                3   0 /lib/x86_64-linux-gnu/libc.so.6\n4788   uname              3   0 /usr/lib/locale/locale-archive\n4795   sed                3   0 /lib/x86_64-linux-gnu/libpcre2-8.so.0\n4794   sed                3   0 /etc/ld.so.cache",
        0),
        (
        "4789   sed                3   0 /lib/x86_64-linux-gnu/libdl.so.2\n4789   sed                3   0 /lib/x86_64-linux-gnu/libpthread.so.0\n4789   sed                3   0 /proc/filesystems\n4789   sed                3   0 /usr/lib/locale/locale-archive\n4789   sed                3   0 /usr/lib/x86_64-linux-gnu/gconv/gconv-modules.cache\n4796   which              3   0 /etc/ld.so.cache\n4796   which              3   0 /lib/x86_64-linux-gnu/libc.so.6\n4796   which              3   0 /usr/bin/which\n4786   make               3   0 .output\n4797   sh                 3   0 /etc/ld.so.cache\n4797   sh                 3   0 /lib/x86_64-linux-gnu/libc.so.6\n4798   rm                 3   0 /etc/ld.so.cache\n4798   rm                 3   0 /lib/x86_64-linux-gnu/libc.so.6\n4798   rm                 3   0 /usr/lib/locale/locale-archive\n4798   rm                 3   0 .output\n4798   rm                 3   0 .output",
        0),
        (
        "4798   rm                -1   2 /usr/share/locale-langpack/en_US.utf8/LC_MESSAGES/libc.mo\n4798   rm                -1   2 /usr/share/locale-langpack/en_US/LC_MESSAGES/libc.mo\n4798   rm                -1   2 /usr/share/locale-langpack/en.UTF-8/LC_MESSAGES/libc.mo\n4798   rm                -1   2 /usr/share/locale-langpack/en.utf8/LC_MESSAGES/libc.mo\n4798   rm                -1   2 /usr/share/locale-langpack/en/LC_MESSAGES/libc.mo\n4798   rm                 3   0 bpf\n4798   rm                 3   0 bpf\n4798   rm                 3   0 pkgconfig\n4798   rm                 3   0 pkgconfig\n4798   rm                 3   0 libbpf\n4798   rm                 3   0 libbpf\n4798   rm                 3   0 staticobjs\n4798   rm                 3   0 staticobjs",
        0),
        (
        "4821   cargo              3   0 /lib/x86_64-linux-gnu/libnettle.so.7\n4821   cargo              3   0 /lib/x86_64-linux-gnu/libgnutls.so.30\n4821   cargo              3   0 /lib/x86_64-linux-gnu/libgssapi_krb5.so.2\n4821   cargo              3   0 /lib/x86_64-linux-gnu/libldap_r-2.4.so.2\n4821   cargo              3   0 /lib/x86_64-linux-gnu/liblber-2.4.so.2\n4821   cargo              3   0 /lib/x86_64-linux-gnu/libbrotlidec.so.1\n4821   cargo              3   0 /lib/x86_64-linux-gnu/libgpg-error.so.0\n4821   cargo              3   0 /lib/x86_64-linux-gnu/libunistring.so.2\n4821   cargo              3   0 /lib/x86_64-linux-gnu/libhogweed.so.5\n4821   cargo              3   0 /lib/x86_64-linux-gnu/libgmp.so.10\n4821   cargo              3   0 /lib/x86_64-linux-gnu/libp11-kit.so.0\n4821   cargo              3   0 /lib/x86_64-linux-gnu/libtasn1.so.6\n4821   cargo              3   0 /lib/x86_64-linux-gnu/libkrb5.so.3\n4821   cargo              3   0 /lib/x86_64-linux-gnu/libk5crypto.so.3",
        0),
        (
        "4821   cargo              8   0 /root/.cargo/registry/cache/index.crates.io-6f17d22bba15001f/aho-corasick-1.1.3.crate\n4821   cargo              9   0 /root/.cargo/registry/src/index.crates.io-6f17d22bba15001f/aho-corasick-1.1.3/.cargo-ok\n4821   cargo              9   0 /root/.cargo/registry/src/index.crates.io-6f17d22bba15001f/aho-corasick-1.1.3/Cargo.toml\n4821   cargo             -1   2 /root/.cargo/registry/src/index.crates.io-6f17d22bba15001f/aho-corasick-1.1.3/src/bin\n4821   cargo             -1   2 /root/.cargo/registry/src/index.crates.io-6f17d22bba15001f/aho-corasick-1.1.3/examples\n4821   cargo             -1   2 /root/.cargo/registry/src/index.crates.io-6f17d22bba15001f/aho-corasick-1.1.3/tests\n4821   cargo             -1   2 /root/.cargo/registry/src/index.crates.io-6f17d22bba15001f/aho-corasick-1.1.3/benches\n4821   cargo              8   0 /root/.cargo/registry/cache/index.crates.io-6f17d22bba15001f/anyhow-1.0.86.crate\n4821   cargo              9   0 /root/.cargo/registry/src/index.crates.io-6f17d22bba15001f/anyhow-1.0.86/.cargo-ok\n4821   cargo              9   0 /root/.cargo/registry/src/index.crates.io-6f17d22bba15001f/anyhow-1.0.86/Cargo.toml\n4821   cargo             -1   2 /root/.cargo/registry/src/index.crates.io-6f17d22bba15001f/anyhow-1.0.86/src/bin\n4821   cargo             -1   2 /root/.cargo/registry/src/index.crates.io-6f17d22bba15001f/anyhow-1.0.86/examples\n4821   cargo              9   0 /root/.cargo/registry/src/index.crates.io-6f17d22bba15001f/anyhow-1.0.86/tests\n4821   cargo             -1   2 /root/.cargo/registry/src/index.crates.io-6f17d22bba15001f/anyhow-1.0.86/benches\n4821   cargo              8   0 /root/.cargo/registry/cache/index.crates.io-6f17d22bba15001f/atty-0.2.14.crate\n4821   cargo              9   0 /root/.cargo/registry/src/index.crates.io-6f17d22bba15001f/atty-0.2.14/.cargo-ok\n4821   cargo              9   0 /root/.cargo/registry/src/index.crates.io-6f17d22bba15001f/atty-0.2.14/Cargo.toml",
        0),
        (
        "4824   cp                 3   0 /lib/x86_64-linux-gnu/libpthread.so.0\n4824   cp                 3   0 /proc/filesystems\n4824   cp                 3   0 /usr/lib/locale/locale-archive\n4824   cp                 3   0 /home/ubuntu/Downloads/project/bpftools/third_party/blazesym/target/release/blazesym.h\n4824   cp                 4   0 /home/ubuntu/Downloads/project/bpftools/memleak/.output/blazesym.h\n4825   sh                 3   0 /etc/ld.so.cache\n4825   sh                 3   0 /lib/x86_64-linux-gnu/libc.so.6\n4826   cc                 3   0 /etc/ld.so.cache\n4826   cc                 3   0 /lib/x86_64-linux-gnu/libc.so.6\n4826   cc                 3   0 /usr/lib/locale/locale-archive\n4826   cc                 3   0 /usr/share/locale/locale.alias\n4826   cc                -1   2 /usr/share/locale/en_US.UTF-8/LC_MESSAGES/gcc-10.mo\n4826   cc                -1   2 /usr/share/locale/en_US.utf8/LC_MESSAGES/gcc-10.mo",
        0),
        (
        "4827   cc1               -1   2 /usr/local/include/limits.h\n4827   cc1               -1   2 /usr/include/x86_64-linux-gnu/limits.h\n4827   cc1                4   0 /usr/include/limits.h\n4827   cc1                4   0 /usr/include/x86_64-linux-gnu/bits/libc-header-start.h\n4827   cc1               -1   2 .output/bits/posix1_lim.h\n4827   cc1               -1   2 /usr/lib/gcc/x86_64-linux-gnu/10/include/bits/posix1_lim.h\n4827   cc1               -1   2 /usr/local/include/bits/posix1_lim.h\n4827   cc1                4   0 /usr/include/x86_64-linux-gnu/bits/posix1_lim.h\n4827   cc1                4   0 /usr/include/x86_64-linux-gnu/bits/wordsize.h\n4827   cc1               -1   2 .output/bits/local_lim.h\n4827   cc1               -1   2 /usr/lib/gcc/x86_64-linux-gnu/10/include/bits/local_lim.h\n4827   cc1               -1   2 /usr/local/include/bits/local_lim.h\n4827   cc1                4   0 /usr/include/x86_64-linux-gnu/bits/local_lim.h\n4827   cc1               -1   2 .output/linux/limits.h",
        0),
        (
        "4828   as                 3   0 /etc/ld.so.cache\n4828   as                 3   0 /lib/x86_64-linux-gnu/libopcodes-2.34-system.so\n4828   as                 3   0 /lib/x86_64-linux-gnu/libbfd-2.34-system.so\n4828   as                 3   0 /lib/x86_64-linux-gnu/libz.so.1\n4828   as                 3   0 /lib/x86_64-linux-gnu/libc.so.6\n4828   as                 3   0 /lib/x86_64-linux-gnu/libdl.so.2\n4828   as                 3   0 /usr/lib/locale/locale-archive\n4828   as                 3   0 .output/memleak.o\n4828   as                 4   0 /tmp/cciAiOT8.s",
        0),
        (
        "5222   go                 7   0 /usr/local/go/src/fmt\n5222   go                 7   0 /usr/local/go/src/fmt/doc.go\n5222   go                 3   0 /usr/local/go/src/errors/errors_test.go\n5222   go                 3   0 /usr/local/go/src/errors/example_test.go\n5222   go                 7   0 /usr/local/go/src/fmt/errors.go\n5222   go                 3   0 /usr/local/go/src/errors/join.go\n5222   go                 3   0 /usr/local/go/src/errors/join_test.go\n5222   go                 7   0 /usr/local/go/src/fmt/errors_test.go\n5222   go                 3   0 /usr/local/go/src/errors/wrap.go\n5222   go                 7   0 /usr/local/go/src/fmt/example_test.go\n5222   go                 3   0 /usr/local/go/src/errors/wrap_test.go\n5222   go                 7   0 /usr/local/go/src/fmt/export_test.go",
        0),
        (
        "5207   install            3   0 /lib/x86_64-linux-gnu/libdl.so.2\n5207   install           -1   2 /home/ubuntu/Documents/libbpf/dest/usr/lib/libpthread.so.0\n5207   install            3   0 /lib/x86_64-linux-gnu/libpthread.so.0\n5207   install            3   0 /proc/filesystems\n5207   install            3   0 /usr/lib/locale/locale-archive\n5207   install            3   0 ../include/uapi/linux/bpf.h\n5207   install            4   0 /home/ubuntu/Documents/libbpf/dest/usr/include/linux/bpf.h\n5207   install            3   0 ../include/uapi/linux/bpf_common.h\n5207   install            4   0 /home/ubuntu/Documents/libbpf/dest/usr/include/linux/bpf_common.h\n5207   install            3   0 ../include/uapi/linux/btf.h\n5207   install            4   0 /home/ubuntu/Documents/libbpf/dest/usr/include/linux/btf.h",
        0),
        (
        "5250   python3            3   0 /usr/lib/python3.8/__pycache__/ssl.cpython-38.pyc\n5250   python3            3   0 /usr/lib/python3.8/lib-dynload/_ssl.cpython-38-x86_64-linux-gnu.so\n5250   python3            3   0 /etc/ld.so.cache\n5250   python3            3   0 /lib/x86_64-linux-gnu/libssl.so.1.1\n5250   python3            3   0 /lib/x86_64-linux-gnu/libcrypto.so.1.1\n5250   python3            3   0 /usr/lib/python3.8/__pycache__/mimetypes.cpython-38.pyc\n5250   python3            3   0 /usr/lib/python3.8/__pycache__/shutil.cpython-38.pyc\n5250   python3            3   0 /usr/lib/python3.8/__pycache__/fnmatch.cpython-38.pyc\n5250   python3            3   0 /usr/lib/python3.8/__pycache__/bz2.cpython-38.pyc\n5250   python3            3   0 /usr/lib/python3.8/__pycache__/_compression.cpython-38.pyc\n5250   python3            3   0 /usr/lib/python3.8/__pycache__/threading.cpython-38.pyc\n5250   python3            3   0 /usr/lib/python3.8/lib-dynload/_bz2.cpython-38-x86_64-linux-gnu.so\n5250   python3            3   0 /etc/ld.so.cache",
        0),
        (
        "5394   debug              3   0 /lib/x86_64-linux-gnu/libnettle.so.8\n5394   debug              3   0 /lib/x86_64-linux-gnu/libgmp.so.10\n5394   debug              3   0 /lib/x86_64-linux-gnu/libkrb5.so.3\n5394   debug              3   0 /lib/x86_64-linux-gnu/libk5crypto.so.3\n5394   debug              3   0 /lib/x86_64-linux-gnu/libcom_err.so.2\n5394   debug              3   0 /lib/x86_64-linux-gnu/libkrb5support.so.0\n5394   debug              3   0 /lib/x86_64-linux-gnu/libsasl2.so.2\n5394   debug              3   0 /lib/x86_64-linux-gnu/libbrotlicommon.so.1\n5394   debug              3   0 /lib/x86_64-linux-gnu/libp11-kit.so.0\n5394   debug              3   0 /lib/x86_64-linux-gnu/libtasn1.so.6\n5394   debug              3   0 /lib/x86_64-linux-gnu/libkeyutils.so.1",
        0),
        (
        "5399   hsts-preload       3   0 /lib/x86_64-linux-gnu/libkeyutils.so.1\n5399   hsts-preload       3   0 /lib/x86_64-linux-gnu/libresolv.so.2\n5399   hsts-preload       3   0 /lib/x86_64-linux-gnu/libffi.so.8\n5399   hsts-preload       3   0 /usr/lib/ssl/openssl.cnf\n5399   hsts-preload       7   0 /etc/nsswitch.conf\n5399   hsts-preload       7   0 /etc/host.conf\n5399   hsts-preload       7   0 /etc/resolv.conf\n5399   hsts-preload       7   0 /etc/hosts\n5399   hsts-preload       7   0 /etc/gai.conf",
        0),
        (
        "23155  touch             -1  36 bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        1),
        (
        "3255   touch             -1  36 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        1),
        (
        "23222  touch             -1  36 hjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkghdnhjfkgh",
        1),
        (
        "23305  cat                4   0 /etc/ld.so.cache\n23305  cat                4   0 /lib/x86_64-linux-gnu/libc.so.6\n23305  cat               -1   2 /path/to/nonexistent/file",
        1),
        (
        "23438  mount              4   0 /etc/ld.so.cache\n23438  mount              4   0 /lib/x86_64-linux-gnu/libmount.so.1\n23438  mount              4   0 /lib/x86_64-linux-gnu/libselinux.so.1\n23438  mount              4   0 /lib/x86_64-linux-gnu/libc.so.6\n23438  mount              4   0 /lib/x86_64-linux-gnu/libblkid.so.1\n23438  mount              4   0 /lib/x86_64-linux-gnu/libpcre2-8.so.0\n23438  mount              4   0 /proc/filesystems\n23438  mount             -1   2 /etc/filesystems\n23438  mount              4   0 /proc/filesystems",
        1),
        (
        "2036   gsd-housekeepin   10   0 /proc/self/mountinfo\n2036   gsd-housekeepin   10   0 /run/mount/utab\n2036   gsd-housekeepin   10   0 /proc/self/mountinfo\n2036   gsd-housekeepin   10   0 /run/mount/utab\n24141  Tutorial_Reduct    4   0 /etc/ld.so.cache\n24141  Tutorial_Reduct    4   0 /lib/x86_64-linux-gnu/libstdc++.so.6\n24141  Tutorial_Reduct    4   0 /lib/x86_64-linux-gnu/libgcc_s.so.1\n24141  Tutorial_Reduct    4   0 /lib/x86_64-linux-gnu/libc.so.6\n24141  Tutorial_Reduct    4   0 /lib/x86_64-linux-gnu/libm.so.6\n2251   gnome-terminal-   26   0 /usr/share/icons/Yaru/scalable/actions/tab-new-symbolic.svg\n2251   gnome-terminal-   26   0 /usr/share/icons/Yaru/scalable/actions/open-menu-symbolic.svg\n2251   gnome-terminal-   26   0 /usr/share/icons/Yaru/scalable/actions/edit-find-symbolic.svg",
        0),
        (
        "359    systemd-journal   51   0 /proc/2019/loginuid\n359    systemd-journal   51   0 /proc/2019/cgroup\n359    systemd-journal   -1   2 /run/systemd/units/log-extra-fields:user@1000.service\n359    systemd-journal   51   0 /proc/2019/comm\n359    systemd-journal   -1   2 /run/log/journal/826b2055389f4f1fba3af26e0a68ca08/system.journal\n359    systemd-journal   51   0 /proc/2019/comm\n359    systemd-journal   -1   2 /run/log/journal/826b2055389f4f1fba3af26e0a68ca08/system.journal\n359    systemd-journal   51   0 /proc/2019/comm\n359    systemd-journal   -1   2 /run/log/journal/826b2055389f4f1fba3af26e0a68ca08/system.journal\n359    systemd-journal   51   0 /proc/2019/comm\n359    systemd-journal   -1   2 /run/log/journal/826b2055389f4f1fba3af26e0a68ca08/system.journal",
        0),
        (
        "739    vmtoolsd           8   0 /etc/mtab\n739    vmtoolsd          11   0 /proc/devices\n739    vmtoolsd           8   0 /sys/class/block/sda5/../device/../../../class\n739    vmtoolsd          -1   2 /sys/class/block/sda5/../device/../../../label\n739    vmtoolsd           8   0 /sys/class/block/sda1/../device/../../../class\n739    vmtoolsd          -1   2 /sys/class/block/sda1/../device/../../../label",
        0),
        (
        "782    irqbalance         6   0 /proc/stat\n739    vmtoolsd           8   0 /etc/mtab\n739    vmtoolsd          11   0 /proc/devices\n739    vmtoolsd           8   0 /sys/class/block/sda5/../device/../../../class\n739    vmtoolsd          -1   2 /sys/class/block/sda5/../device/../../../label\n739    vmtoolsd           8   0 /sys/class/block/sda1/../device/../../../class\n739    vmtoolsd          -1   2 /sys/class/block/sda1/../device/../../../label\n739    vmtoolsd           8   0 /run/systemd/resolve/resolv.conf\n739    vmtoolsd           8   0 /proc/net/route\n739    vmtoolsd           8   0 /proc/net/ipv6_route\n739    vmtoolsd           8   0 /proc/uptime\n1869   gnome-shell       46   0 /proc/self/stat",
        0),
        (
        "2036   gsd-housekeepin   10   0 /run/mount/utab\n2036   gsd-housekeepin   10   0 /proc/self/mountinfo\n2036   gsd-housekeepin   10   0 /run/mount/utab\n24144  Tutorial_Reduct    4   0 /etc/ld.so.cache\n24144  Tutorial_Reduct    4   0 /lib/x86_64-linux-gnu/libstdc++.so.6\n24144  Tutorial_Reduct    4   0 /lib/x86_64-linux-gnu/libgcc_s.so.1\n24144  Tutorial_Reduct    4   0 /lib/x86_64-linux-gnu/libc.so.6\n24144  Tutorial_Reduct    4   0 /lib/x86_64-linux-gnu/libm.so.6\n782    irqbalance         6   0 /proc/interrupts\n782    irqbalance         6   0 /proc/stat\n2251   gnome-terminal-   26   0 /usr/share/icons/Yaru/scalable/actions/tab-new-symbolic.svg\n2251   gnome-terminal-   26   0 /usr/share/icons/Yaru/scalable/actions/open-menu-symbolic.svg\n2251   gnome-terminal-   26   0 /usr/share/icons/Yaru/scalable/actions/edit-find-symbolic.svg",
        0),
        (
        "739    vmtoolsd           8   0 /proc/stat\n739    vmtoolsd           8   0 /proc/zoneinfo\n739    vmtoolsd           8   0 /proc/uptime\n739    vmtoolsd           8   0 /proc/diskstats\n24145  make_circulant     4   0 /etc/ld.so.cache\n24145  make_circulant     4   0 /lib/x86_64-linux-gnu/libstdc++.so.6\n24145  make_circulant     4   0 /lib/x86_64-linux-gnu/libgcc_s.so.1\n24145  make_circulant     4   0 /lib/x86_64-linux-gnu/libc.so.6\n24145  make_circulant     4   0 /lib/x86_64-linux-gnu/libm.so.6\n2030   gsd-color         15   0 /etc/localtime\n2030   gsd-color         15   0 /etc/localtime\n782    irqbalance         6   0 /proc/interrupts\n782    irqbalance         6   0 /proc/stat\n2251   gnome-terminal-   26   0 /usr/share/icons/Yaru/scalable/actions/tab-new-symbolic.svg\n2251   gnome-terminal-   26   0 /usr/share/icons/Yaru/scalable/actions/open-menu-symbolic.svg",
        0),
    ]
    return pd.DataFrame(data, columns=['opensnoop', 'labels'])

# 使用TF-IDF向量化
def vectorize_data(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['opensnoop'])
    y = df['labels']
    return X, y, vectorizer

# 训练模型
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# 预测与评估
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return y_pred

# 定义预测函数
def predict_behavior(model, vectorizer, opensnoop_text):
    opensnoop_vec = vectorizer.transform([opensnoop_text])
    prediction = model.predict(opensnoop_vec)
    return prediction[0]

def read_file(file_path):
    """
    读取文件内容并返回字符串。

    :param file_path: 文件路径
    :return: 文件内容字符串
    """
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def main():
    # 加载数据
    df = load_data()

    # 向量化数据
    X, y, vectorizer = vectorize_data(df)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练模型
    model = train_model(X_train, y_train)

    # 评估模型
    y_pred = evaluate_model(model, X_test, y_test)

    # 示例预测
    # example_opensnoop = "1234   example_process   3   0 /some/path/to/file"

    # 示例预测-badcase
#     example_opensnoop = """23438  mount              4   0 /etc/ld.so.cache
# 23438  mount              4   0 /lib/x86_64-linux-gnu/libmount.so.1
# 23438  mount              4   0 /lib/x86_64-linux-gnu/libselinux.so.1
# 23438  mount              4   0 /lib/x86_64-linux-gnu/libc.so.6
# 23438  mount              4   0 /lib/x86_64-linux-gnu/libblkid.so.1
# 23438  mount              4   0 /lib/x86_64-linux-gnu/libpcre2-8.so.0
# 23438  mount              4   0 /proc/filesystems
# 23438  mount             -1   2 /etc/filesystems
# 23438  mount              4   0 /proc/filesystems
# """
    file_path = 'bt.log'
    example_opensnoop = read_file(file_path)
    
    prediction = predict_behavior(model, vectorizer, example_opensnoop)
    print(f"Predicted Behavior for Example Process: {'Abnormal' if prediction == 1 else 'Normal'}")

if __name__ == "__main__":
    main()
