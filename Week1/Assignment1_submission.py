import sys
program_name = sys.argv[0]
arguments = sys.argv[1:]
start = [] #initialize lists to store the starting and ending indices of circuits
ending = []
lines=[]
if len(arguments) is 1 :        # to check that the no of arguments passed is exactly one
	if (sys.argv[1]).endswith('.netlist') :
		with open(sys.argv[1]) as f : 
			lines = f.readlines()
			k=0
			for i in lines :
				lines[k] = lines[k].split('#')[0]
				lines[k]=lines[k].strip()
				i=lines[k]
				if i == ".circuit":
					start.append(k)	
				elif i == ".end" :
					ending.append(k)
				k=k+1		
			j=len(start) -1
			while j >=0 :
				k = ending[j]-1
				while k>= start[j]+1 :
					rev= lines[k]
					list_res = rev.split()
					m = len(list_res)-1
					while m >=0 :
						print(list_res[m],' ',end='')
						m= m-1
					k = k-1
					print()
				j= j-1
	else :
		print("invalid input text file format Pleae recheck the argument passed")					
else :   # display error message in the situation that incorrect number of parameters are passed
	print("Inappropriate no of parameters passed! Please recheck")	
