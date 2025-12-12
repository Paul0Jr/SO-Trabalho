import sys
import json
import csv
import random
import threading
import time
import hashlib
from queue import Queue
from collections import OrderedDict, defaultdict, deque
from typing import Optional, List, Dict, Tuple
import numpy as np

SIMULATE_SLEEP = False


class CacheBase:
    def __init__(self, capacity: int, policy: str):
        self.capacity = max(0, int(capacity))
        self.policy = policy.upper()
        self.evictions = 0
        self.insertions = 0
        self.lock = threading.RLock()

    def get(self, key: str) -> bool:
        raise NotImplementedError

    def put(self, key: str, dirty: bool = False):
        raise NotImplementedError

    def is_dirty(self, key: str) -> bool:
        raise NotImplementedError

    def mark_clean(self, key: str):
        raise NotImplementedError

    def info(self) -> Dict:
        with self.lock:
            return {"capacity": self.capacity, "policy": self.policy,
                    "evictions": self.evictions, "insertions": self.insertions}


class LRUCache(CacheBase):
    def __init__(self, capacity: int):
        super().__init__(capacity, "LRU")
        self.od = OrderedDict()
        self.dirty = set()

    def get(self, key):
        with self.lock:
            if key in self.od:
                self.od.move_to_end(key)
                return True
            return False

    def put(self, key, dirty=False):
        with self.lock:
            if self.capacity == 0:
                return
            if key in self.od:
                self.od.move_to_end(key)
                if dirty:
                    self.dirty.add(key)
                return
            if len(self.od) >= self.capacity:
                evicted, _ = self.od.popitem(last=False)
                self.dirty.discard(evicted)
                self.evictions += 1
            self.od[key] = True
            if dirty:
                self.dirty.add(key)
            self.insertions += 1

    def is_dirty(self, key):
        with self.lock:
            return key in self.dirty

    def mark_clean(self, key):
        with self.lock:
            self.dirty.discard(key)


class FIFOCache(CacheBase):
    def __init__(self, capacity: int):
        super().__init__(capacity, "FIFO")
        self.queue = deque()
        self.set = set()
        self.dirty = set()

    def get(self, key):
        with self.lock:
            return key in self.set

    def put(self, key, dirty=False):
        with self.lock:
            if self.capacity == 0:
                return
            if key in self.set:
                if dirty:
                    self.dirty.add(key)
                return
            if len(self.queue) >= self.capacity:
                old = self.queue.popleft()
                self.set.remove(old)
                self.dirty.discard(old)
                self.evictions += 1
            self.queue.append(key)
            self.set.add(key)
            if dirty:
                self.dirty.add(key)
            self.insertions += 1

    def is_dirty(self, key):
        with self.lock:
            return key in self.dirty

    def mark_clean(self, key):
        with self.lock:
            self.dirty.discard(key)


class LFUCache(CacheBase):
    def __init__(self, capacity: int):
        super().__init__(capacity, "LFU")
        self.freq = defaultdict(int)
        self.storage = set()
        self.timer = 0
        self.last_access = {}
        self.dirty = set()

    def get(self, key):
        with self.lock:
            if key in self.storage:
                self.freq[key] += 1
                self.timer += 1
                self.last_access[key] = self.timer
                return True
            return False

    def put(self, key, dirty=False):
        with self.lock:
            if self.capacity == 0:
                return

            if key in self.storage:
                self.freq[key] += 1
                self.timer += 1
                self.last_access[key] = self.timer
                if dirty:
                    self.dirty.add(key)
                return

            if len(self.storage) >= self.capacity:
                victim = min(self.storage, key=lambda k: (self.freq[k], self.last_access[k]))
                self.storage.remove(victim)
                del self.freq[victim]
                del self.last_access[victim]
                self.dirty.discard(victim)
                self.evictions += 1

            self.storage.add(key)
            self.freq[key] = 1
            self.timer += 1
            self.last_access[key] = self.timer
            if dirty:
                self.dirty.add(key)
            self.insertions += 1

    def is_dirty(self, key):
        with self.lock:
            return key in self.dirty

    def mark_clean(self, key):
        with self.lock:
            self.dirty.discard(key)


class DeduplicatedCache(CacheBase):
    """Cache with deduplication support for shared pages"""
    def __init__(self, capacity: int, policy: str):
        super().__init__(capacity, policy)
        self.base_cache = make_cache(capacity, policy, enable_dedup=False)
        self.content_hash = {}  # hash -> (file_id, refcount)
        self.file_to_hash = {}  # file_id -> hash
        self.dedup_savings = 0

    def _hash_content(self, file_id: str) -> str:
        return hashlib.md5(file_id.encode()).hexdigest()[:8]

    def get(self, key: str) -> bool:
        with self.lock:
            return self.base_cache.get(key)

    def put(self, key: str, dirty=False):
        with self.lock:
            content_hash = self._hash_content(key)
            
            # Check if content already exists
            if content_hash in self.content_hash:
                self.content_hash[content_hash]['refcount'] += 1
                self.dedup_savings += 1
                return
            
            # New unique content
            self.base_cache.put(key, dirty)
            self.content_hash[content_hash] = {
                'file_id': key,
                'refcount': 1
            }
            self.file_to_hash[key] = content_hash

    def is_dirty(self, key: str) -> bool:
        with self.lock:
            return self.base_cache.is_dirty(key)

    def mark_clean(self, key: str):
        with self.lock:
            self.base_cache.mark_clean(key)

    def info(self) -> Dict:
        base_info = self.base_cache.info()
        base_info['dedup_savings'] = self.dedup_savings
        return base_info


def make_cache(capacity: int, policy: str, enable_dedup: bool = False) -> CacheBase:
    p = policy.upper()
    
    if enable_dedup:
        return DeduplicatedCache(capacity, policy)
    
    if p == "LRU":
        return LRUCache(capacity)
    elif p == "FIFO":
        return FIFOCache(capacity)
    elif p == "LFU":
        return LFUCache(capacity)
    else:
        raise ValueError(f"Unknown cache policy: {policy}. Use FIFO, LRU, or LFU.")


class Disk:
    def __init__(self, latency: int, write_latency: int = None):
        self.latency = int(latency)
        self.write_latency = int(write_latency) if write_latency else self.latency * 2
        self.lock = threading.Lock()
        self.read_count = 0
        self.write_count = 0

    def fetch(self, file_id: str) -> Tuple[str, int]:
        with self.lock:
            self.read_count += 1
        if SIMULATE_SLEEP:
            time.sleep(self.latency / 1000.0)
        return (f"DATA({file_id})", self.latency)

    def write(self, file_id: str, data: str) -> int:
        with self.lock:
            self.write_count += 1
        if SIMULATE_SLEEP:
            time.sleep(self.write_latency / 1000.0)
        return self.write_latency


class Hypervisor:
    def __init__(self, host_cache: CacheBase, host_latency: int, disk: Disk, write_policy: str = "write-through"):
        self.host_cache = host_cache
        self.host_latency = int(host_latency)
        self.disk = disk
        self.write_policy = write_policy
        self.host_hits = 0
        self.disk_fetches = 0
        self.lock = threading.Lock()
        self.contention_time = 0.0
        self.lock_waits = 0

    def fetch(self, file_id: str) -> Tuple[str, int]:
        lock_start = time.time()
        self.host_cache.lock.acquire()
        
        lock_wait = time.time() - lock_start
        if lock_wait > 1e-6:
            with self.lock:
                self.lock_waits += 1
                self.contention_time += lock_wait

        try:
            if self.host_cache.get(file_id):
                with self.lock:
                    self.host_hits += 1
                if SIMULATE_SLEEP:
                    time.sleep(self.host_latency / 1000.0)
                return ("host", self.host_latency)

            content, dlat = self.disk.fetch(file_id)
            self.host_cache.put(file_id, dirty=False)
            with self.lock:
                self.disk_fetches += 1
            return ("disk", dlat)
        finally:
            try:
                self.host_cache.lock.release()
            except RuntimeError:
                pass

    def write(self, file_id: str, data: str) -> Tuple[str, int]:
        """Handle write operations with configurable policy"""
        self.host_cache.lock.acquire()
        
        try:
            if self.write_policy == "write-back":
                # Write to cache only, mark dirty
                self.host_cache.put(file_id, dirty=True)
                if SIMULATE_SLEEP:
                    time.sleep(self.host_latency / 1000.0)
                return ("host", self.host_latency)
            
            elif self.write_policy == "write-through":
                # Write to cache and disk immediately
                disk_lat = self.disk.write(file_id, data)
                self.host_cache.put(file_id, dirty=False)
                if SIMULATE_SLEEP:
                    time.sleep(self.host_latency / 1000.0)
                return ("disk", disk_lat + self.host_latency)
            
            else:  # write-around
                # Write directly to disk, bypass cache
                disk_lat = self.disk.write(file_id, data)
                return ("disk", disk_lat)
                
        finally:
            try:
                self.host_cache.lock.release()
            except RuntimeError:
                pass


class VM:
    def __init__(self, vm_id: int, vm_cache: CacheBase, vm_latency: int, hypervisor: Hypervisor, write_policy: str = "write-through"):
        self.vm_id = vm_id
        self.cache = vm_cache
        self.vm_latency = int(vm_latency)
        self.hypervisor = hypervisor
        self.write_policy = write_policy
        self.accesses = 0
        self.vm_hits = 0
        self.host_hits = 0
        self.disk_hits = 0
        self.writes = 0
        self.reads = 0
        self.latency = 0.0
        self.lock = threading.Lock()

        self.thread: Optional[threading.Thread] = None
        self.request_queue: Queue = Queue()
        self.access_log: List[Dict] = []

    def access(self, file_id: str, operation: str = "read", promote_to_vm: bool = True) -> Tuple[str, int]:
        with self.lock:
            self.accesses += 1
            if operation == "write":
                self.writes += 1
            else:
                self.reads += 1

        # Handle reads
        if operation == "read":
            if self.cache.get(file_id):
                with self.lock:
                    self.vm_hits += 1
                    self.latency += self.vm_latency
                if SIMULATE_SLEEP:
                    time.sleep(self.vm_latency / 1000.0)
                return ("vm", self.vm_latency)

            where, latency = self.hypervisor.fetch(file_id)
            total_latency = latency + self.vm_latency

            with self.lock:
                if where == "host":
                    self.host_hits += 1
                else:
                    self.disk_hits += 1
                if promote_to_vm:
                    self.cache.put(file_id, dirty=False)
                self.latency += total_latency

            if SIMULATE_SLEEP:
                time.sleep(self.vm_latency / 1000.0)

            return (where, total_latency)
        
        # Handle writes
        else:
            if self.write_policy == "write-back":
                if self.cache.get(file_id):
                    self.cache.put(file_id, dirty=True)
                    with self.lock:
                        self.vm_hits += 1
                        self.latency += self.vm_latency
                    return ("vm", self.vm_latency)
                else:
                    where, fetch_lat = self.hypervisor.fetch(file_id)
                    self.cache.put(file_id, dirty=True)
                    total_latency = fetch_lat + self.vm_latency
                    with self.lock:
                        if where == "host":
                            self.host_hits += 1
                        else:
                            self.disk_hits += 1
                        self.latency += total_latency
                    return (where, total_latency)
            
            else:  # write-through or write-around
                where, latency = self.hypervisor.write(file_id, "data")
                total_latency = latency + self.vm_latency
                
                if self.write_policy == "write-through":
                    self.cache.put(file_id, dirty=False)
                
                with self.lock:
                    if where == "host":
                        self.host_hits += 1
                    else:
                        self.disk_hits += 1
                    self.latency += total_latency
                
                return (where, total_latency)

    def process_requests(self):
        thread_name = threading.current_thread().name

        while True:
            item = self.request_queue.get()
            try:
                if item is None:
                    break
                
                step, file_id, operation = item
                where, lat = self.access(file_id, operation)
                
                self.access_log.append({
                    "step": step,
                    "vm": self.vm_id,
                    "file": file_id,
                    "operation": operation,
                    "where": where,
                    "latency": lat,
                    "vm_cache_size": self.cache.capacity,
                    "host_cache_size": self.hypervisor.host_cache.capacity,
                    "thread_id": thread_name
                })
            except Exception as e:
                print(f"[ERRO] VM-{self.vm_id} erro: {e}")
            finally:
                self.request_queue.task_done()

    def start_worker(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(
            target=self.process_requests,
            name=f"VM-{self.vm_id}",
            daemon=False
        )
        self.thread.start()

    def stop_and_join(self, timeout: float = 5.0):
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                print(f"[AVISO] Thread VM-{self.vm_id} não finalizou.")

    def enqueue_access(self, step: Optional[int], file_id: Optional[str], operation: str = "read"):
        if step is None:
            self.request_queue.put(None)
        else:
            self.request_queue.put((step, file_id, operation))

    def stats(self) -> Dict:
        with self.lock:
            return {
                "vm_id": self.vm_id,
                "accesses": self.accesses,
                "reads": self.reads,
                "writes": self.writes,
                "vm_hits": self.vm_hits,
                "host_hits": self.host_hits,
                "disk_hits": self.disk_hits,
                "latency": self.latency,
                "vm_cache_info": self.cache.info()
            }


class Simulator:
    def __init__(self, cfg: Dict):
        random.seed(cfg.get("seed", 0))
        np.random.seed(cfg.get("seed", 0))
        self.cfg = cfg

        write_lat = cfg.get("disk_write_latency", cfg["disk_latency"] * 2)
        self.disk = Disk(cfg["disk_latency"], write_lat)
        
        enable_dedup = cfg.get("enable_deduplication", False)
        self.host_cache = make_cache(cfg["host_cache_size"], cfg["host_cache_policy"], enable_dedup)
        
        write_policy = cfg.get("write_policy", "write-through")
        self.hypervisor = Hypervisor(self.host_cache, cfg["host_latency"], self.disk, write_policy)
        self.vms: List[VM] = []

        for vm_id in range(cfg["vm_count"]):
            vm_cache = make_cache(cfg["vm_cache_size"], cfg["vm_cache_policy"])
            vm = VM(vm_id, vm_cache, cfg["vm_latency"], self.hypervisor, write_policy)
            self.vms.append(vm)

        self.workload = self._build_workload(cfg["workload"])
        self.access_log: List[Dict] = []

    def _build_workload(self, wcfg: Dict) -> List[Dict]:
        mode = wcfg.get("mode", "random")
        
        if mode == "provided":
            return wcfg.get("accesses", [])
        
        elif mode == "zipf":
            return self._build_workload_zipf(wcfg)
        
        elif mode == "random":
            rnd = wcfg.get("random", {})
            length = rnd.get("length", 100)
            files = rnd.get("files", ["A", "B", "C", "D", "E", "F", "G"])
            write_ratio = rnd.get("write_ratio", 0.3)
            
            res = []
            for step in range(length):
                vm = random.randrange(self.cfg["vm_count"])
                file_id = random.choice(files)
                operation = "write" if random.random() < write_ratio else "read"
                res.append({"step": step, "vm": vm, "file": file_id, "operation": operation})
            return res
        else:
            raise ValueError("Unknown workload mode")

    def _build_workload_zipf(self, wcfg: Dict) -> List[Dict]:
        """Generate realistic Zipf-distributed workload"""
        zipf_cfg = wcfg.get("zipf", {})
        length = zipf_cfg.get("length", 1000)
        n_files = zipf_cfg.get("n_files", 100)
        alpha = zipf_cfg.get("alpha", 1.5)
        write_ratio = zipf_cfg.get("write_ratio", 0.3)
        
        ranks = np.arange(1, n_files + 1)
        probs = 1.0 / np.power(ranks, alpha)
        probs /= probs.sum()
        
        workload = []
        for step in range(length):
            vm_id = random.randint(0, self.cfg["vm_count"] - 1)
            file_idx = np.random.choice(n_files, p=probs)
            file_id = f"file_{file_idx}"
            operation = "write" if random.random() < write_ratio else "read"
            
            workload.append({
                "step": step,
                "vm": vm_id,
                "file": file_id,
                "operation": operation
            })
        
        return workload

    def run(self):
        print(f"[SIMULADOR] Iniciando {len(self.vms)} VMs.")
        print(f"[SIMULADOR] Política de escrita: {self.hypervisor.write_policy}")
        start_time = time.time()

        for vm in self.vms:
            vm.start_worker()

        print(f"[SIMULADOR] Distribuindo {len(self.workload)} acessos.\n")
        for op in self.workload:
            step = op.get("step")
            vm_id = int(op["vm"])
            file_id = str(op["file"])
            operation = op.get("operation", "read")
            self.vms[vm_id].enqueue_access(step, file_id, operation)

        for vm in self.vms:
            vm.enqueue_access(None, None)

        for vm in self.vms:
            vm.request_queue.join()

        for vm in self.vms:
            vm.stop_and_join(timeout=5.0)

        for vm in self.vms:
            self.access_log.extend(vm.access_log)
        self.access_log.sort(key=lambda x: (x["step"] if x["step"] is not None else -1, x["vm"]))

        elapsed = time.time() - start_time
        print(f"[SIMULADOR] Tempo de execução: {elapsed:.3f}s")
        print(f"[SIMULADOR] Contenção detectada: {self.hypervisor.lock_waits} esperas por lock")
        print(f"[SIMULADOR] Tempo total em contenção: {self.hypervisor.contention_time*1000:.2f}ms")

    def collect_results(self) -> Dict:
        vms_stats = [vm.stats() for vm in self.vms]
        host_info = self.hypervisor.host_cache.info()
        host_metrics = {
            "host_hits": self.hypervisor.host_hits,
            "disk_fetches": self.hypervisor.disk_fetches,
            "host_cache_info": host_info,
            "lock_contention": {
                "lock_waits": self.hypervisor.lock_waits,
                "total_contention_time_ms": self.hypervisor.contention_time * 1000
            }
        }
        
        total_reads = sum(vm.reads for vm in self.vms)
        total_writes = sum(vm.writes for vm in self.vms)
        
        totals = {
            "total_accesses": len(self.workload),
            "total_reads": total_reads,
            "total_writes": total_writes,
            "total_latency": sum(a["latency"] for a in self.access_log),
            "execution_mode": "concurrent",
            "write_policy": self.hypervisor.write_policy
        }
        return {"vms": vms_stats, "host": host_metrics, "totals": totals, "access_log": self.access_log}

    def save_outputs(self, json_path: Optional[str], csv_path: Optional[str]):
        results = self.collect_results()
        if json_path:
            with open(json_path, "w") as f:
                json.dump(results, f, indent=4)
        if csv_path:
            with open(csv_path, "w", newline="") as f:
                fieldnames = ["step", "vm", "file", "operation", "where", "latency", 
                              "vm_cache_size", "host_cache_size", "thread_id"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in results["access_log"]:
                    row_out = {k: row.get(k, "") for k in fieldnames}
                    writer.writerow(row_out)


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 cache_virtual.py config.json")
        sys.exit(1)
    cfg = load_config(sys.argv[1])
    sim = Simulator(cfg)
    sim.run()
    sim.save_outputs(cfg.get("output_json", "results.json"), cfg.get("output_csv", "results.csv"))


if __name__ == "__main__":
    main()