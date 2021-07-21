
module variables
integer,parameter::N=20000
integer,parameter:: Avg_degree=10 
integer,parameter::half_degree=Avg_degree/2
real(8),parameter::Pi=60./100. !Rewiring
integer,dimension(N*half_degree,2):: connecttable
real(8) p
end module

program WS
use variables;
implicit none
integer i,j,candidate,nconect
integer FirstNode,SecondNode
integer Alt_First,Alt_Second


!Generate the ordered Network 
nconect=0
do i=1,N
do j=1,half_degree
	SecondNode=i+j
	if (SecondNode>N) SecondNode=SecondNode-N
	nconect=nconect+1
	connecttable(nconect,1)=i
	connecttable(nconect,2)=SecondNode
enddo
enddo




!Rewire Attempt
do i=1,nconect
	FirstNode= connecttable(i,1)
	SecondNode= connecttable(i,2)
	p=rand()
	if (p<Pi) then
	!Rewire
		333 continue
		!Choose a candidate for rewiring
		p=rand()
		candidate=int(p*N+1)
		if (candidate==FirstNode) goto 333
	!Test that connection is not repeated
	do j=1,nconect
		if (i==j) cycle
		Alt_First= connecttable(j,1)
		Alt_Second= connecttable(j,2)
	
	if ((FirstNode-Alt_First)**2+(candidate-Alt_Second)**2==0) goto 333
	if ((FirstNode-Alt_Second)**2+(candidate-Alt_Second)**2==0) goto 333
	enddo
	connecttable(i,2)=candidate
	endif
enddo

!Write Matrix
open(10,file="MyMatrix.txt",status="unknown")

do i=1,nconect
write(10,*) connecttable(i,:)
enddo


end program
