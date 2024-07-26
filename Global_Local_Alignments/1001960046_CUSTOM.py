#3
import collections
import numpy as np
class Custom_Alignment:
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
             
            with open("1001960046_D.txt","a") as f:
              for i in range(n+1):
                # print(D[i])
                f.write(str(np.matrix((h[i]))))  
                f.write('\n')
         
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
                      string_seq1+=seq_1[i - 1]
                      string_seq2+=seq_2[j - 1]
                      j -= 1
                  # left
                  else:
                      string_seq1+="_"
                      string_seq2+=seq_2[i - 1]
                      i -= 1
              string_arr.append((string_seq1[::-1], string_seq2[::-1]))
            return string_arr

    
semi_match = 1
miss_match = -1
match = 2
name = "harshinikandimalla"
h_k="abcdefghijklmnopqrstuvwxyz"
chars = set(name)
S={}
for c1 in h_k:
    s={}
    for c2 in h_k:
        if c1==c2:
            s[c2]=match
        else:
            if c1 in chars and c2 in chars:
                s[c2]=semi_match
            else:
                s[c2]=miss_match
    S[c1]=s
print(S)
s=Custom_Alignment()
print(s.local_alignment("harshinikandimalla","thequickbrownfoxjumpsoverthelazydog",S,-2))
with open("10001960046_S.txt","a") as h:
    h.write(str(np.matrix([["" if j == 0 else 
    chr(j+96) for j in range(27)] , 
    ["" if i == 0 else chr(i+96) 
    if chr(i+96) in chars else i 
    if i != 27 
    else "_" for i in range(28)]] + [[chr(i+96)] + list(S[chr(i+96)].values()) 
    for i in range(1, 27)]))) 
