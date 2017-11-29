from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()
instances = [{'city': 'New York'}, {'city': 'San Francisco'}, {'city': 'Chapel Hill'}]
onehot_encoder = vec.fit_transform(instances).toarray()
print(vec.get_feature_names())
print(str(onehot_encoder))

#如何使用DictVertorizer将列表中的字典变成向量的形式