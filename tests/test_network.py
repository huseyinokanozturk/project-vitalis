import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from src.network import Network  
def run_test():
    print("🧪 RHEO Network: Detaylı Sistem Testi Başlıyor...")

    # 1. AĞ KURULUMU
    # 50 Nöron: 10 tanesi Girdi (Input), gerisi Hidden/Output
    net = Network(num_neurons=50, num_inputs=10, num_outputs=5, dt=0.5)

    # Parametreleri testi net görmek için biraz abartalım
    net.energy_cost = 5.0        # Çabuk yorulsunlar
    net.recovery_rate = 0.5      # Orta hızda toparlansınlar
    net.fatigue_factor = 0.5     # Yorgunluk eşiği sert etkilesin

    # 2. SİMÜLASYON DÖNGÜSÜ
    steps = 1000
    
    # Kayıtlar (Log)
    rec_spikes = []          # Raster Plot için (Zaman, Nöron ID)
    rec_voltage_n20 = []     # 20. Nöronun Voltajı
    rec_threshold_n20 = []   # 20. Nöronun Eşiği
    rec_energy_n20 = []      # 20. Nöronun Enerjisi
    rec_dopamine = []        # Ortamdaki Dopamin miktarı

    print(f"⏳ Simülasyon {steps} adım sürecek...")

    for t in range(steps):
        # A. Girdi Oluştur (Sadece ilk 10 nörona)
        # 0 ile 5 arasında rastgele akım
        inputs = np.random.uniform(2, 8, size=10)
        
        # B. Senaryo: DOPAMİN YAĞMURU (Adım 400 ile 600 arası)
        reward_signal = 0.0
        if 400 <= t < 600:
            reward_signal = 1.0  # Yüksek Dopamin! (Heyecan)
        
        # C. Motoru Çalıştır
        # inputs sadece 10 elemanlı, step fonksiyonu bunu içeride hallediyor
        spike_vector = net.step(external_inputs=inputs, reward=reward_signal)

        # D. Veri Kaydetme
        # 1. Raster Plot için kimlerin ateşlediğini bul
        fired_indices = np.where(spike_vector)[0]
        for idx in fired_indices:
            rec_spikes.append((t, idx))
            
        # 2. Tek bir nöronu (Örn: 20. Nöron) mercek altına al
        # (Input nöronu olmayan, içerideki bir nöronu seçtik)
        rec_voltage_n20.append(net.voltages[20])
        rec_threshold_n20.append(net.thresholds[20])
        rec_energy_n20.append(net.energies[20])
        rec_dopamine.append(net.dopamine)

    print("✅ Simülasyon Tamamlandı. Grafikler çiziliyor...")

    # 3. GÖRSELLEŞTİRME
    plt.figure(figsize=(12, 10))

    # Grafik 1: Raster Plot (Tüm Ağın Aktivitesi)
    plt.subplot(3, 1, 1)
    if len(rec_spikes) > 0:
        times, neurons = zip(*rec_spikes)
        plt.scatter(times, neurons, s=2, c='black', alpha=0.6)
    plt.title('Ağ Aktivitesi (Raster Plot)')
    plt.ylabel('Nöron ID')
    plt.axvline(x=400, color='green', linestyle='--', label='Dopamin Başlangıç')
    plt.axvline(x=600, color='red', linestyle='--', label='Dopamin Bitiş')
    plt.legend(loc='upper right')

    # Grafik 2: Seçilen Nöronun Voltaj ve Eşiği
    plt.subplot(3, 1, 2)
    plt.plot(rec_voltage_n20, label='Voltaj (V)', color='blue', alpha=0.5)
    plt.plot(rec_threshold_n20, label='Adaptif Eşik (Th)', color='red', linestyle='--')
    plt.title('Tekil Nöron Dinamiği (Nöron #20)')
    plt.ylabel('mV')
    plt.legend()

    # Grafik 3: Enerji ve Dopamin İlişkisi
    plt.subplot(3, 1, 3)
    plt.plot(rec_energy_n20, label='Enerji (ATP)', color='green')
    plt.plot(np.array(rec_dopamine)*10 + 50, label='Dopamin Sinyali (Ölçeklenmiş)', color='orange', alpha=0.7)
    plt.title('Metabolizma ve Nöromodülasyon')
    plt.xlabel('Zaman (Adım)')
    plt.ylabel('Seviye')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_test()