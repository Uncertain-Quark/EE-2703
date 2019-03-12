import math
import numpy as np
import sys
import cmath

class resistor : 
    def __init__(self,name,node1,node2,value):
        self.name=name
        self.node1=node1
        self.node2=node2
        self.value=value
    def assign(self,matrix,dictionary): 
        matrix[dictionary[self.node1]][dictionary[self.node1]]+=1/(float(self.value))
        matrix[dictionary[self.node1]][dictionary[self.node2]]-=1/(float(self.value))
        matrix[dictionary[self.node2]][dictionary[self.node2]]+=1/(float(self.value))
        matrix[dictionary[self.node2]][dictionary[self.node1]]-=1/(float(self.value))   
        return matrix           
        #it assigns the necessary changes in circuit to be made    

class inductor : 
    def __init__(self,name,node1,node2,value,frequency):
        self.name=name
        self.node1=node1
        self.node2=node2
        self.value=value       
        self.frequency = frequency
        
    def assign(self,matrix,dictionary):
        impedance = 1/complex(0,2*math.pi*float(self.frequency)*float(self.value))
        matrix[dictionary[self.node1]][dictionary[self.node1]]+=impedance
        matrix[dictionary[self.node1]][dictionary[self.node2]]-=impedance
        matrix[dictionary[self.node2]][dictionary[self.node2]]+=impedance
        matrix[dictionary[self.node2]][dictionary[self.node1]]-=impedance
        return matrix

class capacitor : 
    def __init__(self,name,node1,node2,value,frequency):
        self.name=name
        self.node1=node1
        self.node2=node2
        self.value=value       
        self.frequency = frequency
        
    def assign(self,matrix,dictionary):
        impedance = complex(0,2*math.pi*float(self.frequency)*float(self.value))
        matrix[dictionary[self.node1]][dictionary[self.node1]]+=impedance
        matrix[dictionary[self.node1]][dictionary[self.node2]]-=impedance
        matrix[dictionary[self.node2]][dictionary[self.node2]]+=impedance
        matrix[dictionary[self.node2]][dictionary[self.node1]]-=impedance
        return matrix
        
class current_dc :
    def __init__(self,name ,node1,node2,value):
        self.name=name
        self.node1=node1
        self.node2=node2
        self.value=value
        
    def assign(self,dictionary,const):
        const[dictionary[self.node1]]+=float(self.value)
        const[dictionary[self.node2]]-=float(self.value)
        return const                    
        
class current_ac :
    def __init__(self,name,node1,node2,vpp,phase):
        self.name=name
        self.node1=node1
        self.node2=node2
        self.vpp=vpp
        self.phase=phase
        
    def assign(self,const,dictionary):
        value_comp=(float(self.vpp)+0j)*complex(math.cos(float(self.phase)),math.sin(float(self.phase)))/2    
        const[dictionary[self.node1]]+=value_comp
        const[dictionary[self.node2]]-=value_comp
        return const
                
class voltage_dc : 
    def __init__(self,name,node1,node2,value):
        self.name=name
        self.node1=node1
        self.node2=node2
        self.value=value  
    def assign(self,matrix,dictionary,n_nodes,n_vol,const):
        matrix[dictionary[self.node1]][n_nodes+n_vol]=1
        matrix[dictionary[self.node2]][n_nodes+n_vol]=-1
        matrix[n_nodes+n_vol][dictionary[self.node1]]=-1
        matrix[n_nodes+n_vol][dictionary[self.node2]]=1 
        const[n_nodes+n_vol]=float(self.value) 
        return matrix,const    

class voltage_ac :
    def __init__(self,name,node1,node2,vpp,phase):
        self.name=name
        self.node1=node1
        self.node2=node2
        self.vpp=vpp
        self.phase=phase
        
    def assign(self,matrix,dictionary,n_nodes,n_vol,const):
        value_comp=(float(self.vpp)+0j)*complex(math.cos(float(self.phase)),math.sin(float(self.phase)))/2
        matrix[dictionary[self.node1]][n_nodes+n_vol]=1
        matrix[dictionary[self.node2]][n_nodes+n_vol]=-1
        matrix[n_nodes+n_vol][dictionary[self.node1]]=-1 
        matrix[n_nodes+n_vol][dictionary[self.node2]]=1
        const[n_nodes+n_vol]=value_comp
        return matrix,const                    

def find_circuit(lines):
    start_loc,end_loc=0,0
    if '.circuit' in lines:
        start_loc=lines.index('.circuit')
    if '.end' in lines:
        end_loc=lines.index('.end')        
    return start_loc,end_loc

def find_no_of_nodes(lines_loc):
    list_all =set( [j.split()[1] for j in lines]+[j.split()[2] for j in lines])
    return len(list_all),list(list_all)
    #finds the no of nodes so that we can build the matrix    

progname= sys.argv[0]
arguments = sys.argv[1:];frequency=1e-20
components_cir = [];res_cir=[];cap_cir=[];ind_cir=[];vol_cir_ac=[];vol_cir_dc=[];curr_cir_ac=[];curr_cir_dc=[];dict_node={};order=[]

if len(arguments)==1:
    print('valid input')
    with open(sys.argv[1]) as f:
        print('valid file')
        lines = f.readlines()
        lines = [ j.split('#')[0] for j in lines ]
        lines = [ j.strip() for j in lines ]
        start,end=find_circuit(lines)
        try:
            if lines[end+1].split()[0]=='.ac':
                frequency=lines[end+1].split()[2]  
        except Exception :
            frequency=1e-20 
        print(frequency)       
        if start !=0 or end != 0:
            lines=lines[start+1:end]
            for i in range(0,end-start-1):
                if lines[i].split()[0].startswith('R'):
                    res_indiv=resistor(lines[i].split()[0],lines[i].split()[1],lines[i].split()[2],lines[i].split()[3])
                    res_cir.append(res_indiv)
                elif lines[i].split()[0].startswith('V') and (lines[i].split()[3] == 'ac'):
                    vol_indiv=voltage_ac(lines[i].split()[0],lines[i].split()[1],lines[i].split()[2],lines[i].split()[4],lines[i].split()[5])
                    vol_cir_ac.append(vol_indiv)
                elif lines[i].split()[0].startswith('V') and (lines[i].split()[3] == 'dc'):
                    vol_indiv=voltage_dc(lines[i].split()[0],lines[i].split()[1],lines[i].split()[2],lines[i].split()[4])
                    vol_cir_dc.append(vol_indiv)   
                elif lines[i].split()[0].startswith('I') and (lines[i].split()[3] == 'dc'):
                    curr_indiv=current_dc(lines[i].split()[0],lines[i].split()[1],lines[i].split()[2],lines[i].split()[4])    
                    curr_cir_dc.append(curr_indiv)
                elif lines[i].split()[0].startswith('I') and (lines[i].split()[3]=='ac'):
                    curr_indiv=current_ac(lines[i].split()[0],lines[i].split()[1],lines[i].split()[2],lines[i].split()[4],lines[i].split()[5])    
                    curr_cir_ac.append(curr_indiv)
                elif lines[i].split()[0].startswith('C'):
                    cap_indiv=capacitor(lines[i].split()[0],lines[i].split()[1],lines[i].split()[2],lines[i].split()[3],frequency)
                    cap_cir.append(cap_indiv) 
                elif lines[i].split()[0].startswith('L'):
                    ind_indiv=inductor(lines[i].split()[0],lines[i].split()[1],lines[i].split()[2],lines[i].split()[3],frequency)
                    ind_cir.append(ind_indiv)        
            num_nodes,nodes = find_no_of_nodes(lines)
            n = num_nodes
            for i in range(0,len(nodes)) :
                dict_node[nodes[i]]=i
            k=len(vol_cir_ac)+len(vol_cir_dc)   
            matrix_nodal=np.zeros((n+k,n+k),dtype='cfloat')
            b = np.zeros(n+k,dtype='cfloat')
            
            for i in res_cir:
                matrix_nodal=i.assign(matrix_nodal,dict_node)    
            for i in range(0,len(vol_cir_dc)):
                matrix_nodal,b=vol_cir_dc[i].assign(matrix_nodal,dict_node,n,i,b)
                order.append(vol_cir_dc[i].name)
            for i in range(0,len(curr_cir_dc)):
                b=curr_cir_dc[i].assign(dict_node,b)
            for i in range(0,len(vol_cir_ac)):
                matrix,nodal=vol_cir_ac[i].assign(matrix_nodal,dict_node,n,i,b) 
                order.append(vol_cir_ac[i].name)
            for i in cap_cir :
                matrix_nodal=i.assign(matrix_nodal,dict_node)
            for i in ind_cir :
                matrix_nodal=i.assign(matrix_nodal,dict_node)    
            for i in range(0,len(curr_cir_ac)):
                b=curr_cir_ac[i].assign(b,dict_node)    
                
            matrix_nodal[dict_node['GND']]=np.zeros(n+k)
            matrix_nodal[dict_node['GND']][dict_node['GND']]=1
            b[dict_node['GND']]=0
            print('A = ')
            print(matrix_nodal)
            print('B = ')
            print(b)
            ans=np.linalg.solve(matrix_nodal,b)
            ans_pol=[]
            print('Answer = ')
            for i in range(0,n):
                print(nodes[i],' ',(ans[dict_node[nodes[i]]]))
            for i in range(0,k):
                print('current in '+order[i]+' is '+str(ans[n+i])) 
            for i in ans :
                phase = cmath.phase(i)
                mag = abs(i)
                ans_pol.append((mag,phase))
            print('Answer in polar = ')
            for i in range(0,n):
                print(nodes[i],' ',(ans_pol[dict_node[nodes[i]]]))
            for i in range(0,k):
                print('current in '+order[i]+' is '+str(ans_pol[n+i]))     
        else:
            print("no circuit")    
else :
    print('invalid file format')        
