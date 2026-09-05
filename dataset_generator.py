#!/usr/bin/env python3
"""
Synthetic Network Anomaly Detection Dataset Generator
====================================================
Generates a synthetic test dataset mimicking the feature schema and statistical distribution
of the KDD Cup '99 cybersecurity benchmark dataset for model testing and validation.
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, Any, List
import os


class SyntheticDatasetGenerator:
    """Generates synthetic network connection samples with customizable attack ratios."""

    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.attack_types = [
            'normal', 'neptune', 'smurf', 'back', 'teardrop', 'pod',
            'land', 'satan', 'ipsweep', 'portsweep', 'nmap', 'warezclient',
            'warezmaster', 'imap', 'ftp_write', 'multihop', 'phf', 'spy',
            'buffer_overflow', 'loadmodule', 'perl', 'rootkit', 'guess_passwd',
            'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named',
            'xlock', 'xsnoop', 'worm', 'mscan', 'saint', 'apache2', 'udpstorm',
            'processtable', 'mailbomb', 'sqlattack', 'xterm', 'ps'
        ]

        self.protocol_types = ['tcp', 'udp', 'icmp']

        self.services = [
            'ftp_data', 'other', 'private', 'http', 'remote_job', 'name',
            'netbios_ns', 'eco_i', 'mtp', 'telnet', 'finger', 'domain_u',
            'supdup', 'uucp_path', 'Z39_50', 'smtp', 'csnet_ns', 'uucp',
            'netbios_dgm', 'urp_i', 'auth', 'domain', 'ftp', 'bgp', 'ldap',
            'ecr_i', 'gopher', 'vmnet', 'systat', 'http_443', 'efs', 'whois'
        ]

        self.flags = ['SF', 'S0', 'REJ', 'RSTR', 'RSTO', 'RSTOS0', 'RSTRH', 'S1', 'S2', 'S3', 'SH', 'SHR', 'OTH']
        self.difficulties = [15, 16, 17, 18, 19, 20, 21]

    def generate_normal_connection(self) -> Dict[str, Any]:
        """Generate a normal network connection profile."""
        return {
            'duration': np.random.exponential(1.0),
            'protocol_type': np.random.choice(self.protocol_types, p=[0.7, 0.2, 0.1]),
            'service': np.random.choice(self.services[:9], p=[0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1]),
            'flag': np.random.choice(self.flags[:7], p=[0.6, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]),
            'src_bytes': np.random.lognormal(6, 1.5),
            'dst_bytes': np.random.lognormal(6, 1.5),
            'land': 0,
            'wrong_fragment': 0,
            'urgent': 0,
            'hot': np.random.poisson(0.1),
            'num_failed_logins': 0,
            'logged_in': np.random.choice([0, 1], p=[0.3, 0.7]),
            'num_compromised': 0,
            'root_shell': 0,
            'su_attempted': 0,
            'num_root': 0,
            'num_file_creations': 0,
            'num_shells': 0,
            'num_access_files': 0,
            'num_outbound_cmds': 0,
            'is_host_login': 0,
            'is_guest_login': 0,
            'count': np.random.poisson(5),
            'srv_count': np.random.poisson(3),
            'serror_rate': np.random.beta(1, 9),
            'srv_serror_rate': np.random.beta(1, 9),
            'rerror_rate': np.random.beta(1, 9),
            'srv_rerror_rate': np.random.beta(1, 9),
            'same_srv_rate': np.random.beta(8, 2),
            'diff_srv_rate': np.random.beta(2, 8),
            'srv_diff_host_rate': np.random.beta(2, 8),
            'dst_host_count': np.random.poisson(50),
            'dst_host_srv_count': np.random.poisson(20),
            'dst_host_same_srv_rate': np.random.beta(8, 2),
            'dst_host_diff_srv_rate': np.random.beta(2, 8),
            'dst_host_same_src_port_rate': np.random.beta(8, 2),
            'dst_host_srv_diff_host_rate': np.random.beta(2, 8),
            'dst_host_serror_rate': np.random.beta(1, 9),
            'dst_host_srv_serror_rate': np.random.beta(1, 9),
            'dst_host_rerror_rate': np.random.beta(1, 9),
            'dst_host_srv_rerror_rate': np.random.beta(1, 9),
            'class': 'normal',
            'difficulty': np.random.choice(self.difficulties, p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.2])
        }

    def generate_attack_connection(self, attack_type: str) -> Dict[str, Any]:
        """Generate an attack connection profile based on attack category signature."""
        base = self.generate_normal_connection()
        if attack_type == 'neptune':
            base.update({
                'protocol_type': 'tcp', 'flag': 'S0', 'src_bytes': 0, 'dst_bytes': 0,
                'count': np.random.poisson(100), 'srv_count': np.random.poisson(50),
                'serror_rate': 1.0, 'srv_serror_rate': 1.0, 'dst_host_serror_rate': 1.0
            })
        elif attack_type == 'smurf':
            base.update({
                'protocol_type': 'icmp', 'flag': 'SF', 'src_bytes': 28, 'dst_bytes': 0,
                'count': np.random.poisson(200), 'srv_count': np.random.poisson(200),
                'same_srv_rate': 1.0, 'dst_host_same_srv_rate': 1.0
            })
        elif attack_type in ['satan', 'ipsweep', 'portsweep']:
            base.update({
                'count': np.random.poisson(40), 'diff_srv_rate': 1.0, 'dst_host_diff_srv_rate': 1.0
            })
        base['class'] = attack_type
        return base

    def generate_dataset(self, num_samples: int = 10000, attack_ratio: float = 0.3) -> pd.DataFrame:
        """Generate a synthetic dataset DataFrame."""
        print(f"Generating synthetic dataset with {num_samples} samples (Attack ratio: {attack_ratio:.1%})...")
        data = []
        num_attacks = int(num_samples * attack_ratio)
        num_normal = num_samples - num_attacks

        for _ in range(num_normal):
            data.append(self.generate_normal_connection())

        attack_types = [at for at in self.attack_types if at != 'normal']
        probs = [0.2, 0.2, 0.1, 0.1] + [0.4 / (len(attack_types) - 4)] * (len(attack_types) - 4)
        chosen_attacks = np.random.choice(attack_types, size=num_attacks, p=probs)

        for at in chosen_attacks:
            data.append(self.generate_attack_connection(at))

        random.shuffle(data)
        df = pd.DataFrame(data)
        df.insert(0, 'index', range(len(df)))
        return df


def main():
    generator = SyntheticDatasetGenerator(random_seed=42)
    df = generator.generate_dataset(num_samples=10000, attack_ratio=0.3)
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "synthetic_test_dataset.csv")
    df.to_csv(out_file, index=False)
    print(f"Synthetic dataset saved to {out_file}")


if __name__ == "__main__":
    main()
