import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

def plot_hit_rates(metrics: dict):
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        policies = list(metrics.keys())
        x = np.arange(len(policies))
        width = 0.25
        
        vm_rates = [metrics[p]['vm_hit_rate'] for p in policies]
        host_rates = [metrics[p]['host_hit_rate'] for p in policies]
        disk_rates = [metrics[p]['disk_hit_rate'] for p in policies]
        
        rects1 = ax.bar(x - width, vm_rates, width, label='VM Cache', color='#2ecc71')
        rects2 = ax.bar(x, host_rates, width, label='Host Cache', color='#3498db')
        rects3 = ax.bar(x + width, disk_rates, width, label='Disco', color='#e74c3c')
        
        ax.bar_label(rects1, padding=3, fmt='%.1f')
        ax.bar_label(rects2, padding=3, fmt='%.1f')
        ax.bar_label(rects3, padding=3, fmt='%.1f')
        
        ax.set_ylabel('Taxa de Acerto (%)', fontsize=12)
        ax.set_title('Hit Rates por Política', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(policies)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('graph_hit_rates.png', dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"    Erro: {e}")

def plot_latency_comparison(metrics: dict):
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        policies = list(metrics.keys())
        total_lat = [metrics[p]['total_latency'] for p in policies]
        avg_lat = [metrics[p]['avg_latency'] for p in policies]
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
        if len(policies) > len(colors):
            colors = colors * (len(policies) // len(colors) + 1)
        
        ax1.bar(policies, total_lat, color=colors[:len(policies)])
        ax1.set_title('Latência Total (ms)', fontweight='bold')
        for i, v in enumerate(total_lat): 
            ax1.text(i, v, str(int(v)), ha='center', va='bottom')
        
        ax2.bar(policies, avg_lat, color=colors[:len(policies)])
        ax2.set_title('Latência Média por Acesso (ms)', fontweight='bold')
        for i, v in enumerate(avg_lat): 
            ax2.text(i, v, f"{v:.2f}", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('graph_latency.png', dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"    Erro: {e}")

def plot_latency_percentiles(metrics: dict):
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        policies = list(metrics.keys())
        x = np.arange(len(policies))
        width = 0.25
        
        p50 = [metrics[p].get('latency_p50', 0) for p in policies]
        p95 = [metrics[p].get('latency_p95', 0) for p in policies]
        p99 = [metrics[p].get('latency_p99', 0) for p in policies]
        
        rects1 = ax.bar(x - width, p50, width, label='P50 (Mediana)', color='#2ecc71')
        rects2 = ax.bar(x, p95, width, label='P95', color='#f39c12')
        rects3 = ax.bar(x + width, p99, width, label='P99 (Tail)', color='#e74c3c')
        
        ax.bar_label(rects1, padding=3, fmt='%.1f')
        ax.bar_label(rects2, padding=3, fmt='%.1f')
        ax.bar_label(rects3, padding=3, fmt='%.1f')
        
        ax.set_ylabel('Latência (ms)', fontsize=12)
        ax.set_title('Percentis de Latência', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(policies)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('graph_latency_percentiles.png', dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"    Erro: {e}")

def plot_latency_cdf(access_logs: dict):
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
        
        for i, (policy, log) in enumerate(access_logs.items()):
            latencies = sorted([entry['latency'] for entry in log])
            if not latencies:
                continue
            y = np.arange(1, len(latencies) + 1) / len(latencies) * 100
            
            color = colors[i % len(colors)]
            ax.plot(latencies, y, label=policy, linewidth=2, color=color)
        
        ax.set_xlabel('Latência (ms)', fontsize=12)
        ax.set_ylabel('Percentil Acumulado (%)', fontsize=12)
        ax.set_title('CDF de Latência', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('graph_latency_cdf.png', dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"    Erro: {e}")

def plot_read_write_ratio(metrics: dict):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        policies = list(metrics.keys())
        x = np.arange(len(policies))
        width = 0.35
        
        reads = [metrics[p].get('total_reads', 0) for p in policies]
        writes = [metrics[p].get('total_writes', 0) for p in policies]
        
        rects1 = ax.bar(x - width/2, reads, width, label='Leituras', color='#3498db')
        rects2 = ax.bar(x + width/2, writes, width, label='Escritas', color='#e67e22')
        
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        
        ax.set_ylabel('Número de Operações', fontsize=12)
        ax.set_title('Leituras vs Escritas', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(policies)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('graph_read_write_ratio.png', dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"    Erro: {e}")

def plot_evictions(metrics: dict):
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        policies = list(metrics.keys())
        evictions = [metrics[p]['total_evictions'] for p in policies]
        
        bars = ax.bar(policies, evictions, color='#f39c12')
        ax.set_title('Thrashing: Total de Evictions', fontweight='bold')
        ax.set_ylabel('Quantidade de Remoções')
        ax.bar_label(bars)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('graph_evictions.png', dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"    Erro: {e}")

def plot_timeline(log: list, policy: str):
    try:
        steps = [e['step'] for e in log]
        lats = [e['latency'] for e in log]
        colors_map = {'vm': '#2ecc71', 'host': '#3498db', 'disk': '#e74c3c'}
        c_map = [colors_map.get(e['where'], '#000000') for e in log]
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(steps, lats, c=c_map, s=80, alpha=0.7, edgecolors='black')
        ax.set_title(f'Timeline - {policy.upper()}', fontweight='bold')
        ax.set_xlabel('Ordem de Processamento')
        ax.set_ylabel('Latência (ms)')
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#2ecc71', label='VM Hit'),
                           Patch(facecolor='#3498db', label='Host Hit'),
                           Patch(facecolor='#e74c3c', label='Disk Fetch')]
        ax.legend(handles=legend_elements)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'graph_timeline_{policy.lower()}.png', dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"    Erro: {e}")

def plot_efficiency(metrics: dict):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        policies = list(metrics.keys())
        x = np.arange(len(policies))
        
        vm_rates = np.array([metrics[p]['vm_hit_rate'] for p in policies])
        host_rates = np.array([metrics[p]['host_hit_rate'] for p in policies])
        disk_rates = np.array([metrics[p]['disk_hit_rate'] for p in policies])
        
        ax.bar(x, vm_rates, label='VM Cache', color='#2ecc71')
        ax.bar(x, host_rates, bottom=vm_rates, label='Host Cache', color='#3498db')
        ax.bar(x, disk_rates, bottom=vm_rates+host_rates, label='Disco', color='#e74c3c')
        
        ax.set_ylabel('Distribuição (%)')
        ax.set_title('Eficiência de Cache (Distribuição de Acessos)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(policies)
        ax.legend()
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig('graph_efficiency.png', dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"    Erro: {e}")

def plot_contention(metrics: dict):
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        policies = list(metrics.keys())
        
        lock_waits = [metrics[p].get('lock_waits', 0) for p in policies]
        contention_times = [metrics[p].get('contention_time_ms', 0) for p in policies]
        
        ax1.bar(policies, lock_waits, color='#e67e22')
        ax1.set_title('Número de Esperas por Lock', fontweight='bold')
        ax1.set_ylabel('Contagem')
        ax1.grid(axis='y', alpha=0.3)
        if lock_waits and any(lock_waits):
            for i, v in enumerate(lock_waits):
                if v > 0:
                    ax1.text(i, v, str(int(v)), ha='center', va='bottom')
        
        ax2.bar(policies, contention_times, color='#e74c3c')
        ax2.set_title('Tempo Total em Contenção', fontweight='bold')
        ax2.set_ylabel('Tempo (ms)')
        ax2.grid(axis='y', alpha=0.3)
        for i, v in enumerate(contention_times): 
            if v > 0:
                ax2.text(i, v, f"{v:.2f}", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('graph_contention.png', dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"    Erro: {e}")

def plot_throughput(metrics: dict):
    try:
        if not any(metrics[p].get('throughput_ops_sec', 0) for p in metrics.keys()):
            print("    Sem dados de throughput")
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        policies = list(metrics.keys())
        throughput = [metrics[p].get('throughput_ops_sec', 0) for p in policies]
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        bars = ax.bar(policies, throughput, color=colors[:len(policies)])
        
        ax.set_title('Throughput (Operações/Segundo)', fontweight='bold')
        ax.set_ylabel('Ops/seg')
        ax.bar_label(bars, fmt='%.1f')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('graph_throughput.png', dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"    Erro: {e}")

def load_config(path: str) -> Dict:
    with open(path) as f: 
        return json.load(f)

def run_simulation(config: Dict, policy: str, output_suffix: str) -> Dict:
    config['vm_cache_policy'] = policy
    config['host_cache_policy'] = policy
    if 'execution_mode' in config:
        del config['execution_mode']
    
    config['output_json'] = f'results_{output_suffix}.json'
    config['output_csv'] = f'results_{output_suffix}.csv'
    
    temp_config = f'config_temp_{output_suffix}.json'
    with open(temp_config, 'w') as f: 
        json.dump(config, f, indent=4)
    
    try:
        result = subprocess.run(
            [sys.executable, 'cache_virtual.py', temp_config], 
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[ERRO CRÍTICO] A simulação falhou para {policy}.")
        print(e.stderr)
        sys.exit(1)
        
    with open(config['output_json']) as f: 
        results = json.load(f)
    
    try:
        Path(temp_config).unlink()
    except:
        pass
        
    return results

def calculate_metrics(results: Dict) -> Dict:
    if not results['vms']:
        return {}
        
    vm = results['vms'][0] if len(results['vms']) == 1 else {
        'accesses': sum(v['accesses'] for v in results['vms']),
        'reads': sum(v.get('reads', 0) for v in results['vms']),
        'writes': sum(v.get('writes', 0) for v in results['vms']),
        'vm_hits': sum(v['vm_hits'] for v in results['vms']),
        'host_hits': sum(v['host_hits'] for v in results['vms']),
        'disk_hits': sum(v['disk_hits'] for v in results['vms']),
        'vm_cache_info': results['vms'][0]['vm_cache_info']
    }
    
    host = results['host']
    total = vm['accesses']
    
    latencies = [entry['latency'] for entry in results['access_log']]
    
    metrics = {
        'total_accesses': total,
        'total_reads': vm.get('reads', 0),
        'total_writes': vm.get('writes', 0),
        'vm_hits': vm['vm_hits'],
        'vm_hit_rate': (vm['vm_hits'] / total * 100) if total else 0,
        'host_hits': vm['host_hits'],
        'host_hit_rate': (vm['host_hits'] / total * 100) if total else 0,
        'disk_hits': vm['disk_hits'],
        'disk_hit_rate': (vm['disk_hits'] / total * 100) if total else 0,
        'total_evictions': vm['vm_cache_info']['evictions'] + host['host_cache_info']['evictions'],
        'avg_latency': results['totals']['total_latency'] / total if total else 0,
        'total_latency': results['totals']['total_latency'],
        'latency_p50': float(np.percentile(latencies, 50)) if latencies else 0,
        'latency_p95': float(np.percentile(latencies, 95)) if latencies else 0,
        'latency_p99': float(np.percentile(latencies, 99)) if latencies else 0,
        'latency_max': float(max(latencies)) if latencies else 0,
    }
    
    if total > 0:
        metrics['throughput_ops_sec'] = total / (results['totals']['total_latency'] / 1000.0) if results['totals']['total_latency'] > 0 else 0
    
    if 'lock_contention' in host:
        metrics['lock_waits'] = host['lock_contention']['lock_waits']
        metrics['contention_time_ms'] = host['lock_contention']['total_contention_time_ms']
    
    if 'dedup_savings' in host['host_cache_info']:
        metrics['dedup_savings'] = host['host_cache_info']['dedup_savings']
    
    return metrics

def print_clean_table(metrics: Dict[str, Dict]):
    print("\n" + "="*100)
    print("RESULTADOS DA SIMULAÇÃO".center(100))
    print("="*100)
    
    policies = list(metrics.keys())
    headers = "".join([f"{p:>18}" for p in policies])
    print(f"{'Métrica':<30} {headers}")
    print("-"*100)
    
    row_keys = [
        ('VM Hit Rate (%)', 'vm_hit_rate', True, False),
        ('Host Hit Rate (%)', 'host_hit_rate', True, False),
        ('Disk Hit Rate (%)', 'disk_hit_rate', True, False),
        ('Total Reads', 'total_reads', False, False),
        ('Total Writes', 'total_writes', False, False),
        ('Total Evictions', 'total_evictions', False, False),
        ('Latência Total', 'total_latency', False, False),
        ('Latência Média', 'avg_latency', False, True),
        ('Latência P50', 'latency_p50', False, True),
        ('Latência P95', 'latency_p95', False, True),
        ('Latência P99', 'latency_p99', False, True),
        ('Throughput (ops/s)', 'throughput_ops_sec', False, True),
        ('Lock Waits', 'lock_waits', False, False),
        ('Contenção (ms)', 'contention_time_ms', False, True),
        ('Dedup Savings', 'dedup_savings', False, False)
    ]
    
    for label, key, is_pct, is_float in row_keys:
        if not any(metrics[p].get(key, 0) for p in policies):
            continue
            
        row_str = f"{label:<30} "
        for p in policies:
            val = metrics[p].get(key, 0)
            if is_pct:
                row_str += f"{val:>18.1f}"
            elif is_float:
                row_str += f"{val:>18.2f}"
            else:
                row_str += f"{val:>18}"
        print(row_str)
    print("="*100)

def main():
    parser = argparse.ArgumentParser(description="Simulador de Cache (Benchmark)")
    parser.add_argument("config", help="Arquivo de configuração JSON")
    parser.add_argument("--all", action="store_true", help="Executa benchmark comparativo (FIFO, LRU, LFU)")
    
    args = parser.parse_args()
    
    try:
        base_config = load_config(args.config)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{args.config}' não encontrado.")
        sys.exit(1)
    
    if args.all:
        policies = ['FIFO', 'LRU', 'LFU']
        print(f"Modo: Benchmark Comparativo ({', '.join(policies)})")
        
        all_metrics = {}
        access_logs = {}
        
        for policy in policies:
            print(f"\n{'='*60}")
            print(f"SIMULAÇÃO: {policy}".center(60))
            print('='*60)
            results = run_simulation(base_config.copy(), policy, policy.lower())
            all_metrics[policy] = calculate_metrics(results)
            access_logs[policy] = results['access_log']
        
        print_clean_table(all_metrics)
        
        plot_hit_rates(all_metrics)
        plot_latency_comparison(all_metrics)
        plot_latency_percentiles(all_metrics)
        plot_latency_cdf(access_logs)
        plot_read_write_ratio(all_metrics)
        plot_evictions(all_metrics)
        plot_efficiency(all_metrics)
        plot_contention(all_metrics)
        plot_throughput(all_metrics)
        
        for policy in policies:
            plot_timeline(access_logs[policy], policy)
        
    
    else:
        default_policy = base_config.get("vm_cache_policy", "FIFO")
        policies = [default_policy]
        print(f"Modo: Execução Simples ({default_policy})")
        
        all_metrics = {}
        access_logs = {}
        
        print(f"\n{'='*60}")
        print(f"SIMULAÇÃO: {default_policy}".center(60))
        print('='*60)
        results = run_simulation(base_config.copy(), default_policy, default_policy.lower())
        all_metrics[default_policy] = calculate_metrics(results)
        access_logs[default_policy] = results['access_log']
        
        print_clean_table(all_metrics)
        
        plot_hit_rates(all_metrics)
        plot_latency_comparison(all_metrics)
        plot_latency_percentiles(all_metrics)
        plot_latency_cdf(access_logs)
        plot_read_write_ratio(all_metrics)
        plot_evictions(all_metrics)
        plot_efficiency(all_metrics)
        plot_contention(all_metrics)
        plot_throughput(all_metrics)
        plot_timeline(access_logs[default_policy], default_policy)
        
    print("\n Processo concluído.\n")

if __name__ == "__main__":
    main()