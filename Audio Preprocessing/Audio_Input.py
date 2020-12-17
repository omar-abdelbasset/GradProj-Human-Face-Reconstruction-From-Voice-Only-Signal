
import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt

class Audio_Preprocessor:

	def __init__(self, wav_path = "",Processed_path="",sr = 16000, frSize = 512, HopLength = 160, HannWindow = 400):
		# Preprocessing Parameters
		self.Processed_path ="Processed Audio\\"

		if not os.path.exists(self.Processed_path):	
			os.makedirs(self.Processed_path,mode=0o777,exist_ok = False)
		
		self.Sampling_Rate = sr
		self.Frame_Size = frSize
		self.Hop_Length = HopLength
		self.Hann_Window = HannWindow

	def Complex_Spectrogram(self,Audio_Path):

		# load Audio File
		audio, sr = librosa.load(Audio_Path,sr=self.Sampling_Rate,mono=True,duration=6.0)
		# check the Audio duration 
		if(audio.shape[0] < self.Sampling_Rate*6):
			print("Smaller")
			audio = np.resize(audio,(self.Sampling_Rate*6,))
		
		elif(audio.shape[0] > self.Sampling_Rate*6):
			print("bigger")
			audio = audio[0:self.Sampling_Rate*6-1]
		
		# calculate the STFT
		Frequency_Components = librosa.stft(audio, n_fft = self.Frame_Size, hop_length = self.Hop_Length, win_length = self.Hann_Window, window = "hann",center=False )

		print(Frequency_Components.shape)

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

	def write(self, Real_Part, Imaginary_Part, Name):
		
		output_Path = self.Processed_path + Name + ".txt"
		#with open(output_Path,"w+") as output:
		np.savetxt(output_Path,np.r_[Real_Part,Imaginary_Part])

	def read(self, Name):
		input_Path = Name + ".txt"
		real=[]
		imaginary=[]
		counter = 0
		with open(input_Path,"r") as file:
			for line in file:
				if(counter<257):
					real.append(np.float_(line.split(' ')))
				else:
					imaginary.append(np.float_(line.split(' ')))
				counter= counter+1
		return real,imaginary

# Testing
Preprocessor = Audio_Preprocessor()

# Put the Audio File Path
Audio_Path = "E:\\GraduationProject\\Code\\AVDataset\\audios\\_2Rjfow5gow.wav"

Real_Part,Imaginary_Part =Preprocessor.Complex_Spectrogram(Audio_Path)

print(Real_Part.shape,Imaginary_Part.shape)
Preprocessor.Draw_Spectrograms(Real_Part,Imaginary_Part)




