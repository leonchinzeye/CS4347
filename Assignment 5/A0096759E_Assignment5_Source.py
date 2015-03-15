import scipy.io.wavfile
import scipy.signal
import scipy.fftpack
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#Prepare data structures for analysis
results_buffered = []

#Declare hamming array to save construction time
hamming_array = []
for i in range(0,1290):
    hamming_array.append(scipy.signal.hamming(1024))
hamming_array = np.array(hamming_array)

#Declare plot colours
plot_colours = ['blue','green','red','cyan','magenta','gold','black']

def main():
    paths_list = get_path_list('music_speech.mf')
    filterbank = compute_mel_filterbank()
    analyse_wave_files(paths_list, filterbank)
    write_arff_file_buffered()

def get_path_list(path):
    # Read music_speech.mf and get list of paths to the wav file
    paths_file = open(path, 'rb')
    paths_list = paths_file.readlines()
    paths_file.close()
    return paths_list

def analyse_wave_files(paths_list, filterbank):
    # For each wav file, run the analysis
    for path_class in paths_list:
#         path_class = paths_list[0]
        path, file_class = path_class.split("\t")
        file_class = file_class.strip()
        rate, data = scipy.io.wavfile.read(path)
        data = data/32768.0        
        analyse_buffered(data,file_class,filterbank)

def analyse_buffered(data,file_class,filterbank):
    #Get buffers
    buffers = []
    for i in range(0,len(data)/512):
        start = i*512
        end = start+1024
        if end < len(data):
            buffers.append(data[start:end])
    buffers = np.array(buffers)
    print buffers[0]
    buffers_diff = np.diff(buffers) #y(t) = x(t) - x(t-1) + 0.05*x(t-1)
    buffers_diff = buffers[:,:-1]*0.05 + buffers_diff
    temp = buffers
    buffers = np.zeros((1290,1024))
    buffers[:,0:1] = temp[:,0:1]
    buffers[:,1:] = buffers_diff
    buffers = buffers * hamming_array
    print buffers[0]
    fft_array = np.abs(np.delete(scipy.fftpack.fft(buffers),np.r_[len(buffers[0])/2+1:len(buffers[0])],1))
    
    #calculate weighted sums for each filterbank and do a DCT
    weighted_sums = np.dot(fft_array,filterbank)
    weighted_sums = np.log10(weighted_sums)
    weighted_sums = scipy.fftpack.dct(weighted_sums)
    
    #calculate mean and std for each bin
    mean_vector = np.mean(weighted_sums, axis=0)
    std_vector = np.std(weighted_sums, axis=0)
    result_vector = []
    for item in mean_vector:
        result_vector.append(item)
    for item in std_vector:
        result_vector.append(item)
    result_vector.append(file_class)
    results_buffered.append(result_vector)
    
def compute_mel_filterbank():
    #Compute Mel Filterbank with min freq = 0 and max freq = 22050 (sampling rate)/2
    num_filters = 26
    min_mel = 0.0
    max_mel = to_mel(22050.0/2)
    mel_points = []
    freq_points = []
    for i in range(0,num_filters+2):
        mel_points.append(min_mel + i*((max_mel-min_mel)/(num_filters+1)))
    for mel in mel_points:
        freq_points.append(from_mel(mel))
    
    #Create filterbank matrix
    filterbank = []
    for i in range(0,num_filters):
        left = np.floor(freq_points[i]*1024/22050)
        top = np.round(freq_points[i+1]*1024/22050)
        right = np.ceil(freq_points[i+2]*1024/22050)
        left_pad = [0]*(left)
        middle_part = []
        for j in range(0,int(top-left)+1):
            middle_part.append(j/(top-left))
        for j in range(0,int(right-top)):
            middle_part.append((right-top-j-1)/(right-top))
        right_pad = [0]*(512-right)
        temp = left_pad + middle_part + right_pad
        filterbank.append(temp)
    filterbank = np.transpose(np.array(filterbank))
    save_graphs(filterbank)
    
    return filterbank

def to_mel(freq):
    return 1127.0*np.log(1+(freq/700.0))
    
def from_mel(mel):
    return (np.exp(mel/1127.0) - 1) * 700.0
    
def write_arff_file_buffered():
    # Print header into output_buffered.arff
    output_file = open('output_buffered.arff', 'wb')
    header = "@RELATION music_speech\n"
    for i in range(0,(len(results_buffered[0])-1)/2):
        header += "@ATTRIBUTE BIN_" + str(i) + "_MEAN NUMERIC\n"
    for i in range(0,(len(results_buffered[0])-1)/2):
        header += "@ATTRIBUTE BIN_" + str(i) + "_STD NUMERIC\n"
    header += "@ATTRIBUTE class {music,speech}\n\n"
    header += "@DATA\n"
    output_file.write(header)

    # Print results into output.arff
    data = ""
    for result in results_buffered:
        for i in range(0,len(result)-1):
            data += str("%.6f" % result[i]) + ","
        data += str(result[len(result)-1]) + "\n"
    output_file.write(data)
    output_file.close()


def save_graphs(filterbank):
    filterbank = np.transpose(filterbank)
    x_axis = np.r_[0:11025:22050/1024.0, 11025]
    
    plt.clf()
    for i in range(0,len(filterbank)):
        plt.plot(x_axis, filterbank[i], c=plot_colours[i%7])
    set_plot_labels(plt)
    plt.savefig("filterbank_whole.png")
    
    plt.clf()
    for i in range(0,len(filterbank)):
        plt.plot(x_axis, filterbank[i], marker='o', c=plot_colours[i%7])
    plt.xlim([0,300])
    set_plot_labels(plt)
    plt.savefig("filterbank_0-300.png")
    
def set_plot_labels(plt):
    plt.title('26 Triangular MFCC Filters, 22050Hz signal, window size 1024')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    
main()