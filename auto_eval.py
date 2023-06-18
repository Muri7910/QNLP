from statistics import mean

paths = ['/Users/jakobmurauer/PycharmProjects/discopy_0.5.0/experiment_mul_cl/my_doku_3/logs/30.txt',
         '/Users/jakobmurauer/PycharmProjects/discopy_0.5.0/experiment_mul_cl/my_doku_3/logs/31.txt',
         '/Users/jakobmurauer/PycharmProjects/discopy_0.5.0/experiment_mul_cl/my_doku_3/logs/32.txt',
         '/Users/jakobmurauer/PycharmProjects/discopy_0.5.0/experiment_mul_cl/my_doku_3/logs/33.txt',
         '/Users/jakobmurauer/PycharmProjects/discopy_0.5.0/experiment_mul_cl/my_doku_3/logs/34.txt']

batch_high_train_acc = []
batch_high_dev_acc = []

for dok in paths:
    dev_acc = []
    train_acc = []
    with open(dok) as f:
        for line in f:
            if line.startswith("Epoch"):
                train_acc.append(float(line[65:71]))
                dev_acc.append(float(line[85:]))
        batch_high_train_acc.append(max(train_acc))
        batch_high_dev_acc.append(max(dev_acc))

print('The highest dev acc overall is:', max(batch_high_dev_acc))
print('The mean of the highest dev_acc is:', mean(batch_high_dev_acc))
print('The mean of the highest train_acc is:', mean(batch_high_train_acc))



