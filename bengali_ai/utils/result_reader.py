import matplotlib.pyplot as plt

# File Path
filePath = "/Users/imrankabir/Desktop/research/bengali_ocr_app/bengali_ocr/bengali_ai/logs/resultData.txt"

with open(filePath, 'r') as f:
    lines = f.readlines()

# training reasult
# 4 training result store variable
data_dict_train = {
    'consonant accuracy': [],
    'vowel accuracy': [],
    'root accuracy': [],
    'all accuracy': []
}

dashes_list = [
    [4, 2, 4, 2],
    [2, 2, 2, 2],
    [4, 2, 2, 4],
    [3, 1, 3, 1]
]


# reading file
for i, line in enumerate(lines):
    if line.startswith("Epoch"):
        line_train = line.replace("\n", "").split()
        consonant_accuracy = line_train[16][:-1]
        data_dict_train['consonant accuracy'].append(float(consonant_accuracy))
        vowel_accuracy = line_train[19][:-1]
        data_dict_train['vowel accuracy'].append(float(vowel_accuracy))
        root_accuracy = line_train[22][:-1]
        data_dict_train['root accuracy'].append(float(root_accuracy))
        all_accuracy = line_train[25][:-1]
        data_dict_train['all accuracy'].append(float(all_accuracy))

# test result
# 4 test result store variable
data_dict_val = {
    'consonant accuracy': [],
    'vowel accuracy': [],
    'root accuracy': [],
    'all accuracy': []
}

# reading file
for i, line in enumerate(lines):
    if line.startswith("Test"):
        line_train = line.replace("\n", "").split()
        consonant_accuracy = line_train[13][:-1]
        data_dict_val['consonant accuracy'].append(float(consonant_accuracy))
        vowel_accuracy = line_train[16][:-1]
        data_dict_val['vowel accuracy'].append(float(vowel_accuracy))
        root_accuracy = line_train[19][:-1]
        data_dict_val['root accuracy'].append(float(root_accuracy))
        all_accuracy = line_train[22][:-1]
        data_dict_val['all accuracy'].append(float(all_accuracy))

X = []
Y = []
labels = []

plot_dict = data_dict_train # use data_dict_train for validation plot

for i, type_ in enumerate(plot_dict.keys()):
    x = []
    y = []
    counter = 0
    for j in range(0, len(plot_dict[type_]), 22):
        counter += 1
        x.append(counter)
        y.append(plot_dict[type_][j])
    X.append(x)
    Y.append(y)
    labels.append(type_)

fig, ax = plt.subplots()

for k in range(len(Y)):
    ax.plot(X[k], Y[k], label=labels[k])

plt.xlabel("Accuracy")
plt.ylabel("Epoch")
plt.title("Accuracy vs Epoch")
plt.xticks(rotation=0)


plt.legend()
# plt.legend(ncol=2, handleheight=0.005, labelspacing=0.002, bbox_to_anchor=(1.0, 1.0), prop={'size': 7})

plt.show()
