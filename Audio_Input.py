
import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt

class Audio_Preprocessor:

	# Preprocessing Parameters
	Sampling_Rate = 16000
	Frame_Size = 512
	Hop_Length = 160
	Hann_Window = 400

	def Complex_Spectrogram(self,Audio_Path):

		# load Audio File
		audio, sr = librosa.load(Audio_Path,sr=self.Sampling_Rate,mono=True,duration = 6.0)

		# calculate the STFT
		Frequency_Components = librosa.stft(audio, n_fft = self.Frame_Size, hop_length = self.Hop_Length, win_length = self.Hann_Window )

		# seperate the Real and imaginary parts and apply the Power low
		Real_Part      = np.real(Frequency_Components)
		Imaginary_Part = np.imag(Frequency_Components)

		Real_Part      = np.sign(Real_Part) * ( np.abs(Real_Part) ** 0.3 )
		Imaginary_Part = np.sign(Imaginary_Part) * ( np.abs(Imaginary_Part) ** 0.3 )

		return Real_Part,Imaginary_Part


	def Draw_Spectrograms(self,Real_Part,Imaginary_Part):
		fig = plt.figure(figsize=(25, 10))

		ax1=fig.add_subplot(121)
		librosa.display.specshow(librosa.power_to_db(Real_Part),sr=self.Sampling_Rate, hop_length=self.Hop_Length,x_axis="time",y_axis="log")

		ax1.set_title('Real')

		ax2=fig.add_subplot(122)

		ax2.set_title('Imaginary')

		librosa.display.specshow(librosa.power_to_db(Imaginary_Part),sr=self.Sampling_Rate, hop_length=self.Hop_Length,x_axis="time",y_axis="log")

		plt.colorbar(format="%+2.f")
		plt.show()



# Testing
Preprocessor = Audio_Preprocessor()

# Put the Audio File Path
Audio_Path = " "

Real_Part,Imaginary_Part =Preprocessor.Complex_Spectrogram(Audio_Path)
Preprocessor.Draw_Spectrograms(Real_Part,Imaginary_Part)

