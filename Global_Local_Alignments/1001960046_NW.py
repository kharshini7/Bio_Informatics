import numpy as np
class Solution:
    #1
        def global_alignment(self, sequence_A: str, sequence_B:str,substitution: dict, gap: int ):
            n = len(sequence_A)
            m = len(sequence_B)    
            traceback = [['' for j in range(m+1)] for i in range(n+1)]

            if n * m == 0:
                return n + m

            
            d = []
            for i in range(n + 1):
                 row = [0] * (m + 1)
                 d.append(row)
        
            for i in range(1,n + 1):
                d[i][0] = i*gap
                traceback[i][0] = 'l'
            for j in range(1,m + 1):
                d[0][j] = j*gap
                traceback[0][j] = 'u'

            # compute DP
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    left = d[i - 1][j] + gap
                    down = d[i][j - 1] + gap
                    left_down = d[i - 1][j - 1] + substitution[sequence_A[i-1]][sequence_B[j-1]]                    
                    d[i][j] = max(left, down, left_down)
                    if d[i][j] == left_down:
                        traceback[i][j] += 'd'
                    if d[i][j] == left:
                        traceback[i][j] += 'l'
                    if d[i][j] == down:
                        traceback[i][j] += 'u'
            print(d)
            ans=self.global_traceback(traceback,sequence_A,sequence_B,n,m)
            return ans
        
        def global_traceback(self, traceback, seq1, seq2, i, j, stri_seq1='', stri_seq2=''):
            if i == 0 and j == 0:
                return [(stri_seq1, stri_seq2)]
            alignments = []
            for direction in traceback[i][j]:
                if direction == 'd':
                    N_W_A = seq1[i-1] + stri_seq1
                    N_W_B = seq2[j-1] + stri_seq2
                    alignments += self.global_traceback(traceback, seq1, seq2, i-1, j-1, N_W_A, N_W_B)
                elif direction == 'l':
                    N_W_A = seq1[i-1] + stri_seq1
                    N_W_B = '-' + stri_seq2
                    alignments += self.global_traceback(traceback, seq1, seq2, i-1, j, N_W_A, N_W_B)
                elif direction == 'u':
                    N_W_A = '-' + stri_seq1
                    N_W_B = seq2[j-1] + stri_seq2
                    alignments += self.global_traceback(traceback, seq1, seq2, i, j-1, N_W_A, N_W_B)
            return alignments
        

d = {'a': {'a':1,'t':-1,'c':-1,'g':-1}, 't':
{'a':-1,'t':1,'c':-1,'g':-1}, 'c':
{'a':-1,'t':-1,'c':1,'g':-1}, 'g':
{'a':-1,'t':-1,'c':-1,'g':1}}
ans=Solution()
ans1 = ans.global_alignment("gata","ctac",d,-2)
print(ans1)