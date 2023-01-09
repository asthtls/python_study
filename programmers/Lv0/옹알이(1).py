from itertools import permutations

def solution(babbling):
    answer = 0
    words = ['aya', 'ye','woo','ma']
    return_answer = []
    
    for i in range(1, len(words)+1):
        for j in permutations(words, i):
            return_answer.append(''.join(j))
    
    for i in babbling:
        if i in return_answer:
            answer+=1
    
    return answer