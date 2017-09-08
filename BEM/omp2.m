function xr=omp2(A, y, s_num, ep)

	% initialization
	r0=y;
	S=[];
	xr=[];

	stop=0;
	num=0;
	while (stop==0)

		num=num+1;
		val_seq=r0'*A;
		val_seq=abs(val_seq);
		[m, idx]=max(val_seq);
		S=[S idx];
		A_block=A(:,S);
		xh=inv(A_block'*A_block)*A_block'*y;
		
		r0_new=y-A_block*xh;

		if (num==s_num) 
			xr=zeros(size(A, 2), 1);
			xr(S, 1)=xh;
			break;
		elseif (norm(r0_new)>norm(r0))
			if (num==1) 
				xr=zeros(size(A,2), 1);
				xr(S, 1)=xh;
			end
			break;
		elseif (norm(r0_new)<ep)
			xr=zeros(size(A, 2), 1);
			xr(S, 1)=xh;
			break;
		else
			xr=zeros(size(A, 2), 1);
			xr(S, 1)=xh;
		end

		r0=r0_new;
		
	end

end
