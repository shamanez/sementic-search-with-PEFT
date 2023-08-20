import datasets

dataset = datasets.load_dataset("smangrul/amazon_esci")

print(dataset)

#pos = dataset["train"].filter(lambda example: example['relevance_label']==1)
neg = dataset["train"].filter(lambda example: example['relevance_label']==0)

#print(pos)
print("=========")
print(neg)