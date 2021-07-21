module variables
!Network parameters 
	integer,parameter::  NeuronN= 20000 !=N=Number of Neurons= Network size
	integer,parameter:: Avg_degree=10 !=<k>
	integer,parameter:: Nbonds=NeuronN*Avg_degree/2
!Time Parameters
	integer,parameter:: n_s=50000    
	integer,parameter::     MeassTime=0.9*n_s
	integer,parameter::   transient=0.1*n_s
!Model Parameters
	real(8),parameter:: r1= 0.001  !Spontaneous activation 
	real(8),parameter:: r2= 0.3 !Refractory to Quiescent probability n which is active)
	real(8),parameter:: lambda=12.5
!Threshold Variables
	real(8)::  T  ! Threshold T
	real(8),parameter::  Delta_T=0.005 ! Threshold Change
	real(8),parameter:: T_0=0.15
	real(8),parameter:: T_F=0.65
!Neuron States and  Neighbor lists
	integer S(1:NeuronN) !Neuron State
	integer S_prev(1:NeuronN) !Previous Neuron State
	integer,parameter:: x1=5*Avg_degree
	integer,dimension(NeuronN):: MyNeighbors=0 ! MyNeighbors(i) = Ammount of neighbors of i
	integer,dimension(NeuronN,x1):: NeighborN=0 ! NeighborN(i,k)= identity of the k-tjh neighbor of i
	real(8),dimension(NeuronN,x1):: Wij=0 ! Wij(i,k)= weight of the k-tjh connetcion of i, i.e., the conection with  NeighborN(i,k)
	real(8)::weight    
!OBSERVABLES		
	integer:: activity,prev_activity
	real(8):: AVGact,AVGact2,AvgActPrev
	real(8):: f_s, varf_s,AC1  !fraction of actives, variance of activity and first autocorrelation coefficient
!AUXILIAR VARIABLES
	integer it !iteration step
	integer i,j, jj,k
	real(8) p !for random numbers
end module

program GH1
use variables
implicit none   

!Load connection matrix and generate weights
open(99,file= "../Matrix/MyMatrix.txt",status="old",action="read")
do k=1,Nbonds
read(99,*) i,j
p=rand(); weight=-log(p)/lambda

MyNeighbors(i)=MyNeighbors(i)+1
NeighborN(i,MyNeighbors(i))=j
Wij(i,MyNeighbors(i))=weight
				
MyNeighbors(j)=MyNeighbors(j)+1
NeighborN(j,MyNeighbors(j))=i
Wij(j,MyNeighbors(j))=weight				
enddo	    
close(99)

!Random Initial Condition
do i=1,NeuronN;	p=rand(); S(i)= int(p*3); enddo
! RUN


open(101,file= "Results.txt",status="unknown")
do T = T_0,T_F,Delta_T
call BlockStep
enddo !T 
close(101)

! RUN Backwards 
open(101,file= "BackResults.txt",status="unknown")
do T = T_F,T_0,-Delta_T
call BlockStep
enddo !T 
close(101)

end program


subroutine BlockStep
use variables;
implicit none

!Transient
do it=1,transient;call step;enddo
!Now we compute variables
avgact=0 !Average Activity
avgact2=0 !Average Activity**2
AvgActPrev=0 !Average Activity(t)*Activity(t-1)

do it=1,MeassTime;
prev_activity=activity;
call step 

avgact=avgact+activity
avgact2=avgact2+activity**2
AvgActPrev=AvgActPrev+activity*prev_activity

enddo
avgact=avgact/real(measstime)
avgact2=avgact2/real(measstime)
AvgActPrev=AvgActPrev/real(measstime)

f_s= avgact/real(NeuronN)
varf_s= real(avgact2-avgact**2)/real(NeuronN)**2
AC1= real((AvgActPrev-avgact**2)/(avgact2-avgact**2))

write(101,*) real(T ),f_s, varf_s, AC1
write(*,*) real(T ),f_s, varf_s, AC1
end subroutine

!------------------------------------------------
subroutine step
!Generates new State from previous state
use variables;
implicit none
real(8) sum_transmit

S_prev=S
Activity=0
do i=1,NeuronN
!Previously active -> Refractory Always
	if (S_prev(i).eq.1) then; S(i)=2  ; endif
!Refractary becomes Quiescent with probability r_2       
	if (S_prev(i).eq.2)then
	 p=rand()
		if (p<r2) then; S(i)=0;
		else;  S(i)=2 ;  endif               
	endif
!Previously Quiescent- > Excited 
	if (S_prev(i).eq.0)then
		s(i)=0
! Spontaneous Activation     
		p=rand()
		if (p<r1)then
			S(i)=1;     
		else;  
! Transmitted activation
		sum_transmit=0; 
		do k=1,MyNeighbors(i)
			j=NeighborN(i,k)
			if (S_prev(j)==1) sum_transmit=sum_transmit+Wij(i,k)
		end do           
		if(sum_transmit > T ) S(i)=1
        end if
	endif
! Now we count active neurons        
if (S(i)==1) then
activity=activity+1
!if (i<301) write(*,*) it,i
endif
end do !i
end subroutine
