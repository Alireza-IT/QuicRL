import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use("seaborn-v0_8-white")

def sketching_plot(path,plt_label,xlabel,ylabel,plt_title):
    data_list = []
    data= []
    x_axis =[]
    with open(path) as file:
        for line in file: 
            line = line.strip() 
            data_list.append(line) 
        for i in range(0,len(data_list)):
            # ages_x.append(i)
            data.append(data_list[0])
        plt.plot(x_axis, data,label=plt_label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(plt_title)
        plt.legend()
        plt.savefig(f"Plot of {plt_title}")
        
