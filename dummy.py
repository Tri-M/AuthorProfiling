lsa_scores = [
    [[0.0014294843355880326, 0.027215256643738764, 0.09921987619951231, 0.24461301520111137, 0.008828115740309539],
     [0.006660090721934835, 0.054172298251357845, 0.1351884164010011, 0.17400595629029691, -0.0293772990119534]],
    # Add more LSA scores for other users as needed
]

user_labels = [1, 2]  # User labels corresponding to the LSA scores

# Convert LSA scores and user labels into feature vectors and labels
X = [tweet for user in lsa_scores for tweet in user]
y = user_labels

print(X)

print(y)