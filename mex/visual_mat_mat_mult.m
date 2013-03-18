clear all
M = csvread('output1.csv',0,0);
%#Minibatch Size, Attribute Size, Number of elements, create matrices gpu [ms], create matrices cpu[ms], fill_random gpu[ms], load kernel[ms], gpu mult[ms], load data[ms], cpu mult[ms]
m = sort(unique(M(:,1)));
a = sort(unique(M(:,2)));
whos
%%
close all

  subplot(1,2,1)
t =[];
for T = 1:10
for N = 1:length(m)
    M1 = rand(m(N),a(floor(end/2)));
    M2 = rand(a(floor(end/2)),1024);
    clear M3;
    tic
    M3 = M1*M2;
    t(T,N) = toc;
end
end
t=t*1000;
t = mean(t);  
  
  
idx = M(:,2) == a(floor(end/2));
plot(M(idx,1),M(idx,end-3)*1000,'r')
hold on
plot(M(idx,1),M(idx,end-1)*1000,'b')
plot(M(idx,1),t,'g')
ylabel('ms');
xlabel('minibatch size');
title({'Mat-Mat-mult simulation',
     [num2str(a(floor(end/2))) ' Attributes, Layer 1 1024 perceptions']});
  legend({'GPU','EIGEN','MATLAB'});
 
  subplot(1,2,2) 
t =[];
for T = 1:10
for N = 1:length(m)
    M1 = rand(m(N),a(end));
    M2 = rand(a(end),1024);
    clear M3;
    tic
    M3 = M1*M2;
    t(T,N) = toc;
end
end
t=t*1000;
t = mean(t);


idx = M(:,2) == a(end);
plot(M(idx,1),M(idx,end-3)*1000,'r')
hold on
plot(M(idx,1),M(idx,end-1)*1000,'b')
plot(M(idx,1),t,'g')
ylabel('ms');
xlabel('minibatch size');
title({'Mat-Mat-mult simulation',
      '2032 Attributes, Layer 1 1024 perceptions'});
  legend({'GPU','EIGEN','MATLAB'});



%%
figure(3)
[X,Y] = meshgrid(m,a)
t = []
for aa = 1:length(a)
    for mm = 1:length(m)
        clear M3
        M1 = rand(m(mm),a(aa));
        M2 = rand(a(aa),1024);
        tic
        M3 = M1*M2;
        t(aa,mm) = toc;
    end
end
t = t *1000;
subplot(1,2,1)
surfc(X,Y,reshape(M(:,end-3),length(a),length(m))*1000);
subplot(1,2,2)
surfc(X,Y,t)
