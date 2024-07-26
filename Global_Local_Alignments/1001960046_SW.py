
import numpy as np
class Solution:
    
        #2
        def local_alignment(self, sequence_A: str, sequence_B:str, substitution: dict, gap: int ):
            n = len(sequence_A)
            m = len(sequence_B)

            # if the string is empty
            if n * m == 0:
                return n + m

            h = []
            for i in range(n + 1):
                 row = [0] * (m + 1)
                 h.append(row)

            i = 0
            j = 0
            while i <= n:
               h[i][0] = 0
               i += 1
            while j <= m:
               h[0][j] = 0
               j += 1

            # compute DP
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    left = h[i - 1][j] + gap
                    down = h[i][j - 1] + gap
                    left_down = h[i - 1][j - 1] + substitution[sequence_A[i-1]][sequence_B[j-1]]                    
                    h[i][j] = max(left, down, left_down,0)   
            print(h)    
            return self.local_traceback(h,sequence_A,sequence_B,gap)
        
        def find_maximum_coordinates(self,score_matrix):
          score=np.array(score_matrix)
          result = np.where(score_matrix == np.amax(score_matrix))
          number_maximum_coordinates=result[0].size
          maximum_coords_arr=[]
          for i in range(number_maximum_coordinates):
            maximum_coords = (result[0][i], result[1][i])
            maximum_coords_arr.append(maximum_coords)
          
          return maximum_coords_arr
    
        def local_traceback(self,score_matrix, seq_1, seq_2, gap):
            
            res = self.find_maximum_coordinates(score_matrix)
           
            string_arr=[]
            for k in range(len(res)):
              i=res[k][0]
              j=res[k][1]
              j-=1
              i-=1
              
              string_seq1=""
              string_seq2=""
              string_seq1+=seq_1[i]
              string_seq2+=seq_2[j]
              
              while score_matrix[i][j] != 0:
                  diagonal = score_matrix[i - 1][j - 1]
                  up = score_matrix[i - 1][j]
                  left = score_matrix[i][j - 1]
                  maximum_points = max(0,diagonal, up, left)
                  score_matrix[i][j] = 0
                 
                  if i > 0 and j > 0 and diagonal == maximum_points:
                      string_seq1+=seq_1[i - 1]
                      string_seq2+=seq_2[j - 1]
                      i -= 1
                      j -= 1
                  # up
                  elif j > 0 and up == maximum_points:
                      string_seq1+="_"
                      string_seq2+=seq_2[j - 1]
                      j -= 1
                  # left
                  else:
                      string_seq1+="_"
                      string_seq2+=seq_2[i - 1]
                      i -= 1
              string_arr.append((string_seq1[::-1], string_seq2[::-1]))
            return string_arr



h = {'a': {'a':1,'t':-1,'c':-1,'g':-1}, 't':
{'a':-1,'t':1,'c':-1,'g':-1}, 'c':
{'a':-1,'t':-1,'c':1,'g':-1}, 'g':
{'a':-1,'t':-1,'c':-1,'g':1}}
ans=Solution()
ans=ans.local_alignment("gata","ctac",h,-2)
print(ans)


