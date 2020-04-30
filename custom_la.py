import os
import pickle
import numpy as np
import subprocess
import time

class Distr_LA:
  host_file = "hosts"
  num_of_process = 1
  def __init__ (self):
    pass

  @classmethod
  def configure(self, host_file="hosts", num_of_process = 1):
    self.host_file = host_file
    self.num_of_process = num_of_process
  
  @classmethod
  def distr_gauss(self, a, b, num_of_process = None):
    if num_of_process == None:
      cmd = "bash rs.sh " + self.host_file + " init.py " + str(self.num_of_process) + " gauss"
    else:
      cmd = "bash rs.sh " + self.host_file + " init.py " + str(num_of_process) + " gauss"
    
    with open('gauss_a', 'wb') as f:
      pickle.dump(a, f)

    with open('gauss_b', 'wb') as f:
      pickle.dump(b, f)

    # os.system(cmd)
    # print("cmd: ", cmd)
    # subprocess.Popen(cmd)
    FNULL = open(os.devnull, 'w')
    subprocess.call(cmd, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
    # process.wait()
    # print("return code: ", process.returncode)


    with open('gauss_x', 'rb') as f:
      x = pickle.load(f)

    os.remove('gauss_a')
    os.remove('gauss_b')
    os.remove('gauss_x')

    # return solution x
    return x


  @classmethod
  def distr_mult(self, a, b, num_of_process = None):
    if num_of_process == None:
      cmd = "bash rs.sh " + self.host_file + " init.py " + str(self.num_of_process) + " mul"
    else:
      cmd = "bash rs.sh " + self.host_file + " init.py " + str(num_of_process) + " mul"
    with open('mult_a', 'wb') as f:
      pickle.dump(a, f)

    with open('mult_b', 'wb') as f:
      pickle.dump(b, f)

    # os.system(cmd)
    FNULL = open(os.devnull, 'w')
    subprocess.call(cmd, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
    # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    # process.wait()

    with open('mult_ab', 'rb') as f:
      x = pickle.load(f)

    os.remove('mult_a')
    os.remove('mult_b')
    os.remove('mult_ab')

    # return solution x
    return x


    
  @classmethod
  def distr_det(self, a, num_of_process = None):
    if num_of_process == None:
      cmd = "bash rs.sh " + self.host_file + " init.py " + str(self.num_of_process) + " det"
    else:
      cmd = "bash rs.sh " + self.host_file + " init.py " + str(num_of_process) + " det"
    with open('det_a', 'wb') as f:
      pickle.dump(a, f)

    #  use this to print the background stout
    # os.system(cmd)
    FNULL = open(os.devnull, 'w')
    subprocess.call(cmd, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
    # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    # process.wait()

    with open('det_ans', 'rb') as f:
      x = pickle.load(f)

    os.remove('det_a')
    os.remove('det_ans')

    # return solution x
    return x



  @classmethod
  def distr_inverse(self, a, b, num_of_process = None):
    if num_of_process == None:
      cmd = "bash rs.sh " + self.host_file + " init.py " + str(self.num_of_process) + " inverse"
    else:
      cmd = "bash rs.sh " + self.host_file + " init.py " + str(num_of_process) + " inverse"
    with open('inverse_a', 'wb') as f:
      pickle.dump(a, f)

    #  use this to print the background stout
    # os.system(cmd)
    FNULL = open(os.devnull, 'w')
    subprocess.call(cmd, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
    # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    # process.wait()

    with open('inverse_ans', 'rb') as f:
      x = pickle.load(f)

    os.remove('inverse_a')
    os.remove('inverse_ans')

    # return solution x
    return x


  @classmethod
  def distr_transpose(self, a, num_of_process = None):
    if num_of_process == None:
      cmd = "bash rs.sh " + self.host_file + " init.py " + str(self.num_of_process) + " transpose"
    else:
      cmd = "bash rs.sh " + self.host_file + " init.py " + str(num_of_process) + " transpose"

    with open('transpose_a', 'wb') as f:
      pickle.dump(a, f)

    FNULL = open(os.devnull, 'w')
    subprocess.call(cmd, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
    # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    # process.wait()
    # print("return code: ", process.returncode)
    # os.system(cmd)

    with open('transposed_a', 'rb') as f:
      x = pickle.load(f)

    os.remove('transpose_a')
    os.remove('transposed_a')

    # return solution x
    return x



  @classmethod
  def distr_qr(self, a, num_of_process = None):
    if num_of_process == None:
      cmd = "bash rs.sh " + self.host_file + " init.py " + str(self.num_of_process) + " qr"
    else:
      cmd = "bash rs.sh " + self.host_file + " init.py " + str(num_of_process) + " qr"

    with open('qr_decompose_a', 'wb') as f:
      pickle.dump(a, f)

    FNULL = open(os.devnull, 'w')
    subprocess.call(cmd, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
    # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    # process.wait()
    # os.system(cmd)

    with open('qr_decomposed_q', 'rb') as f:
      q = pickle.load(f)
    with open('qr_decomposed_r', 'rb') as f:
      r = pickle.load(f)

    os.remove('qr_decompose_a')
    os.remove('qr_decomposed_q')
    os.remove('qr_decomposed_r')

    # return solution x
    return q, r


  def checkDiagonal(arr):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if i == j:
                continue
            else:
                if abs(arr[i][j]) > 0.001:
                    return False
    return True


  def printLambda(self, arr):
      count = 1
      ei_vals = []
      for i in range(len(arr)):
          for j in range(len(arr[i])):
              if(i == j):
                  temp = arr[i][j]
                  if(abs(temp) < 0.000000000001):
                      temp = 0
                  # print("Lamda"+str(count) +": " + str(temp))
                  ei_vals.append(temp)
                  count += 1

        
      with open("ei_values", 'wb') as f:
        pickle.dump(ei_vals, f)
  
  def qrFactorization(self, arr, num_of_process):
      temp = arr
      i = 0
      prev_q = arr

      while(True):
          Q, R = self.distr_qr(temp, num_of_process)
          # print('QR: ', Q, R)
          if i == 0:
            prev_q = Q
          else:
            prev_q = np.dot(prev_q, Q)

          temp = np.dot(R, Q)
          # print("temp: ", temp)
          if(self.checkDiagonal(temp)):
              # print("Number of Factorizations: " + str(i+1))
              break
          else:
              i += 1
      
      with open('ei_vectors', 'wb') as f:
        pickle.dump(prev_q, f)

      # print("eigen vectors: ", prev_q)
      return temp



  @classmethod
  def distr_ei(self, a, num_of_process=None):
    self.printLambda(self, self.qrFactorization(self, a, num_of_process))

    with open('ei_values', 'rb') as f:
      ei_val = pickle.load(f)

    with open('ei_vectors', 'rb') as f:
      ei_vector = pickle.load(f)

    # os.remove('ei_a')
    os.remove('ei_values')
    os.remove('ei_vectors')

    
    return ei_val, ei_vector

