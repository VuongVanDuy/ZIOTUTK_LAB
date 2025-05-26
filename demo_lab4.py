import numpy as np
import matplotlib.pyplot as plt
import pyttsx3
import soundfile as sf
import scipy.signal as signal
import librosa

# Hàm tạo tiếng ồn trắng
def generate_white_noise(duration, sample_rate):
    return np.random.randn(duration * sample_rate)

# Hàm tạo giọng nói từ văn bản
def generate_speech(text, language='en', gender='male'):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    voice = voices[0] if gender == 'male' else voices[1]  # Chọn giọng nam hoặc nữ
    engine.setProperty('voice', voice.id)
    engine.setProperty('rate', 150)  # Tốc độ nói
    engine.save_to_file(text, 'speech.wav')
    engine.runAndWait()

# Hàm tính toán phổ tín hiệu
def plot_spectrum(signal_data, sample_rate):
    f, Pxx = signal.welch(signal_data, sample_rate, nperseg=1024)
    plt.figure(figsize=(10, 6))
    plt.semilogy(f, Pxx)
    plt.title('Ampitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True)
    plt.show()

# Tính toán tỷ lệ tín hiệu trên tiếng ồn (SNR)
def calculate_snr(signal, noise):
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Thực thi chương trình
sample_rate = 44100  # Tần số mẫu
duration = 5  # Thời gian tín hiệu (giây)

# Tạo tiếng ồn trắng
white_noise = generate_white_noise(duration, sample_rate)
# Lưu tiếng ồn trắng vào tệp
sf.write('white_noise.wav', white_noise, sample_rate)

# Tạo giọng nói từ văn bản
generate_speech("Hello, how are you?", language='en', gender='female')

# Đọc tệp âm thanh
speech, _ = sf.read('speech.wav')

#########################################################
# Đọc file âm thanh (giữ nguyên sample rate gốc)
y, sr = librosa.load("speech.wav", sr=None)

# Tăng biên độ sóng âm (hệ số >1 để tăng, <1 để giảm)
factor = 50.0  # tăng gấp đôi biên độ
y_amplified = y * factor

# Ghi ra file WAV mới với dữ liệu đã khuếch đại
sf.write("output_librosa.wav", y_amplified, sr)

##############################################################

# Kết hợp tiếng ồn với giọng nói
noisy_signal = speech + 0.5 * white_noise[:len(speech)]  # 0.5 là tỷ lệ mức độ tiếng ồn

# Vẽ phổ của giọng nói và tín hiệu có tiếng ồn
plot_spectrum(speech, sample_rate)
plot_spectrum(noisy_signal, sample_rate)

# Tính toán tỷ lệ tín hiệu trên tiếng ồn (SNR)
snr = calculate_snr(speech, white_noise[:len(speech)])
print(f"Tỷ lệ tín hiệu trên tiếng ồn (SNR): {snr} dB")
