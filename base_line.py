

p_file=open('data/BaseLine/negative_words.txt',encoding='utf-8')
n_file=open('data/BaseLine/positive_words.txt',encoding='utf-8')

p_list=p_file.read().split('\n')
#print(p_list)
n_list=n_file.read().split('\n')
#print(n_list)

t_file=open('data/BaseLine/test_sentence.txt',encoding='utf-8')
text=t_file.read().split('\n')

for line in text:
    p=0
    n=0
    for word in line.split():
        if word in p_list:
            p+=1
        if word in n_list:
            n+=1
    if p>n:
        res='pos'
    else:
        res='neg'

    print (res)