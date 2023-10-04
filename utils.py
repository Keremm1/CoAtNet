import os
import matplotlib.pyplot as plt

def count_files_in_directory(path):
    list_of_folder = os.listdir(path)
    number_of_files = len(list_of_folder)
    return number_of_files



def plot_acc(train_accuracy, val_accuracy,
             accuracy, top_5_accuracy,
             save_name, save_path="graphs"):

    epochs = range(1, len(train_accuracy) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.axhline(y=accuracy, color='g', linestyle='--', label='Test Accuracy')
    plt.legend()

    plt.axhline(y=top_5_accuracy, color='y', linestyle='--', label='Top 5 Test Accuracy')
    plt.legend()
    
    plt.savefig(f'{save_path}/{save_name}_{count_files_in_directory(save_path)}.png')
    plt.show()

