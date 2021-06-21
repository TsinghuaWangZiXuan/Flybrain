from matplotlib import pyplot as plt

losses = []
with open('./results/training_history_2021-5-10.txt', 'r') as file:
    for line in file:
        line = line.strip().split()
        if line[0] == 'Train':
            continue
        print(line)
        losses.append(float(line[-1]))

plt.figure()
plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('Training Loss')
plt.show()