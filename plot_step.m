%%
% the chosen model (4 state, subseries =200)
%% learned 2 state model  subseries=50
% validation loss  1.1903176307678223
A=[0.25892714  0.02125828;-0.6919281   0.17042749];
B=[-0.21627925 -0.0987151   0.06778879;-0.354768    0.09810929 -0.07273055];
C=[0.29548362 -0.56777394];
D=0;
ts=5;
sys = ss(A,B,C,D,ts);
step(sys)
%% learned 2 state model  subseries=400
% validation loss  1.1030536890029907
A=[0.14053252,-0.36566183;-0.46399364,-0.31118467];
B=[0.22369824,0.3010773,-0.07944702;-0.20917645,-0.22528267,-0.0524387];
C=[0.4431089,-0.56475943];
D=0;
ts=5;
sys = ss(A,B,C,D,ts);
step(sys)

%% learned 3 state model subseries=50
% validation loss  1.1199733018875122
A=[-0.20468305 -0.15762964 -0.27268916;
    -0.48475    -0.12887418 -0.04021144;
    -0.43825504  0.37917134 -0.04751418];
B=[0.10975306  0.04732624 -0.24841562;
    0.16496366  0.14728484  0.34937814;
    -0.32426608  0.12106289 -0.19939707];
C=[-0.16595249 -0.5456792  -0.48140964];
D=0;
ts=5;
sys = ss(A,B,C,D,ts);
step(sys)

%% learned 3 state model subseries=400
% validation loss  1.1811745166778564
A=[-0.35908377  0.1070694  -0.14868103;
    -0.0884248  -0.2296993   0.0999795;
    0.06164128  0.3222499   0.24255393];
B=[0.36767304 -0.1703708  -0.23183373;
    0.09809564  0.02941514  0.31707227;
    0.24076413 -0.02929946 -0.23198277];
C=[-0.68066764 -0.12430349 -0.38992277];
D=0;
ts=5;
sys = ss(A,B,C,D,ts);
step(sys)

%% learned 4 state model subseries=50
% validation loss  1.1477457284927368
A=[0.10219794 -0.305089    0.21109945 -0.03365075;
    -0.58049536  0.1622826  -0.1609498   0.35896778;
    0.36601692  0.18221109 -0.03277422  0.20404571;
    0.49393523 -0.04505786  0.20074409 -0.136552];
B=[-0.02606489 -0.01776552  0.02458843;
    0.3606792  -0.11092178  0.01919511;
    0.2760311   0.09368141 -0.17933297;
    -0.18156989  0.17096153 -0.07306829];

C=[0.17880835 -0.35622355  0.12305978  0.5038483];
D=0;
ts=5;
sys = ss(A,B,C,D,ts);
step(sys)

%% learned 4 state model subseries=200
% validation loss  1.1599960327148438
A=[-0.06824921  0.00085048 -0.00838581 -0.0172115;
    -0.53445333  0.21884347 -0.3806465   0.0174005;
    0.02541124 -0.02280101 -0.17882209  0.20034575;
   -0.55188805  0.23825075 -0.3449853   0.3421386];
B=[0.04751976 -0.3038336   0.11502537;
    -0.24056381 -0.12971258 -0.19785942;
    0.05863554 -0.06726048  0.35304385;
    0.3333706  -0.23543009 -0.14228687];

C=[0.07582217 -0.47991046 -0.11649413 -0.38406187];
D=0;
ts=5;
sys = ss(A,B,C,D,ts);
step(sys)

%% learned 4 state model subseries=400
% validation loss  1.1866322755813599
A=[-0.43905374  0.06132863  0.05446228 -0.35001898;
    0.46658573  0.00780277  0.38548642 -0.37989187;
    0.20968322 -0.19270487  0.09823235 -0.3023902;
    -0.1726205   0.05577308 -0.24330449 -0.07580545];
B=[-0.14548731 -0.18562779  0.2921172;
    0.01339952  0.33271277 -0.01750143;
    0.2587358  -0.19702607 -0.02251318;
    0.01654192 -0.55053246 -0.18912724];

C=[-0.35211483  0.2442789  -0.00211679 -0.35765955];
D=0;
ts=5;
sys = ss(A,B,C,D,ts);
step(sys)

%% learned 5 state model subseries=400
% validation loss  1.1803869009017944
% training loss is: 0.16955173589910072
A=[ 0.41777533 -0.06000374  0.13804418  0.06812168  0.09363393;
    0.00453269 -0.18981078 -0.22396232 -0.11870068  0.02205453;
    0.05242376  0.1493878   0.41774005  0.09217896 -0.17706113;
    -0.25951907  0.29848534 -0.02949474 -0.4110325   0.02489619;
    -0.39602193  0.09679861  0.2896758  -0.1608829  -0.17463621];
B=[-0.13420923 -0.23230562  0.10961781;
    -0.04943481 -0.12555744  0.0891491;
    -0.16450992 -0.15723637 -0.40364745;
    0.0482797   0.4221597   0.509698;
    0.23200303 -0.3157489   0.03759441];

C=[-0.03721095 -0.34263378  0.31894368 -0.33041885  0.11422286];
D=0;
ts=5;
sys = ss(A,B,C,D,ts);
step(sys)


%% learned 6 state model subseries=400
% validation loss  1.1295784711837769
% training loss is: 0.3109994228929281
A=[ 0.23185733  0.00960041  0.2615965   0.27612084 -0.0134261  -0.05756565;
    0.35280496 -0.21767597  0.09735207  0.01551503  0.15299062 -0.16015437;
    -0.03298337 -0.14454637  0.2476859  -0.35644522 -0.016909   -0.01861048;
    -0.10856585 -0.18542153  0.23424928 -0.01141369 -0.09250638 -0.00778373;
    -0.3208722   0.04322042 -0.1459052   0.00581772 -0.11941451 -0.10303479;
    0.2718441  -0.13541266 -0.13926347 -0.26033854  0.01885658  0.28602257];
B=[0.24809706 -0.17277537  0.29378352;
   -0.36133778 -0.2278098   0.10885836;
   -0.05199601 -0.02011568  0.01451185;
   -0.31920385 -0.25697258 -0.04586912;
   -0.23498219  0.35141242  0.07774526;
   -0.11987062  0.06741718 -0.17573209];

C=[0.38409245  0.32162327 -0.14439029  0.19117723 -0.07591133  0.01222935];
D=0;
ts=5;
sys = ss(A,B,C,D,ts);
step(sys)