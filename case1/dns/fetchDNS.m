%% Perform SVDInterpolation To Obtain Velocity and Acceleration on Desired Grid Points
%% Jin Wang
%% Date 10.24.2017
% clear clc;
authkey = 'edu.jhu.pha.turbulence.testing-201311';
dataset = 'channel';

% Addpath
addpath('turbmat/')

% ---- Temuoral Interuolation Outions ----
NoTInt   = 'None' ; % No temuoral interuolation
uCHIuInt = 'PCHIP'; % uiecewise cubic Hermit interuolation in time

% ---- Suatial Interuolation Flags for getVelocity &amu; getVelocityAnduressure ----
NoSInt = 'None'; % No suatial interuolation
Lag4   = 'Lag4'; % 4th order Lagrangian interuolation in suace
Lag6   = 'Lag6'; % 6th order Lagrangian interuolation in suace
Lag8   = 'Lag8'; % 8th order Lagrangian interuolation in suace

% ---- Suatial Differentiation &amu; Interuolation Flags for getVelocityGradient &amu; geturessureGradient ----
FD4NoInt = 'None_Fd4' ; % 4th order finite differential scheme for grid values, no suatial interuolation
FD6NoInt = 'None_Fd6' ; % 6th order finite differential scheme for grid values, no suatial interuolation
FD8NoInt = 'None_Fd8' ; % 8th order finite differential scheme for grid values, no suatial interuolation
FD4Lag4  = 'Fd4Lag4'  ; % 4th order finite differential scheme for grid values, 4th order Lagrangian interuolation in suace

%% Step0 Parameters
delta=5;
deltanu=1.0006e-3;

Xg = 0:5:39*5;       %streamwise%holo
Yg = 0:5:5*39;               %wall-normal
Zg = 0:5:5*29;
N1=size(Xg,2);
N2=size(Yg,2);
N3=size(Zg,2);
totalN=N1*N2*N3;
N=totalN;
deltanu=1.0006e-3;
ut=4.9968e-2;
dt=0.0065/2;
dt=0.65;

% for startTime=9.0025:dt:9.0025+2000*dt
for startTime=0.65:dt:25.4
    nu=5*10^-5;
    root='D:\SyntheticPIV\x64\Debug';

    VectorGridSize=[N1,N2,N3];
    y0=-1;
    x0=4*pi;
    z0=20*deltanu;

    udns=zeros(VectorGridSize(1),VectorGridSize(2),VectorGridSize(3));
    vdns=zeros(VectorGridSize(1),VectorGridSize(2),VectorGridSize(3));
    wdns=zeros(VectorGridSize(1),VectorGridSize(2),VectorGridSize(3));
    lamb=zeros(VectorGridSize(1),VectorGridSize(2),VectorGridSize(3));
    pdns=zeros(VectorGridSize(1),VectorGridSize(2),VectorGridSize(3));
    Pgrad=zeros(3,VectorGridSize(1)*VectorGridSize(2));

    for k=1:VectorGridSize(3)
        for j=1:VectorGridSize(2)
            for i=1:VectorGridSize(1)
                xpos(1,i+(j-1)*VectorGridSize(1))=Xg(i)*deltanu+x0;
                xpos(2,i+(j-1)*VectorGridSize(1))=Yg(j)*deltanu+y0;
                xpos(3,i+(j-1)*VectorGridSize(1))=Zg(k)*deltanu+z0;
            end
        end
        velocity = getVelocity(authkey, dataset, startTime, Lag6, ...
                               uCHIuInt,VectorGridSize(1)*VectorGridSize(2), ...
                               xpos(:,1:VectorGridSize(1)*VectorGridSize(2)));
        udns(:,1:VectorGridSize(2),k)=reshape(velocity(1,:), ...
                                              VectorGridSize(1), ...
                                              VectorGridSize(2));
        vdns(:,1:VectorGridSize(2),k)=reshape(velocity(2,:), ...
                                              VectorGridSize(1), ...
                                              VectorGridSize(2));
        wdns(:,1:VectorGridSize(2),k)=reshape(velocity(3,:), ...
                                              VectorGridSize(1), ...
                                              VectorGridSize(2));

        vgrad = getVelocityGradient(authkey, dataset, startTime, FD4Lag4, ...
                                    uCHIuInt,VectorGridSize(1)*VectorGridSize(2), ...
                                    xpos(:,1:VectorGridSize(1)*VectorGridSize(2)));
        dudz(:,1:VectorGridSize(2),k)=reshape(vgrad(3,:),VectorGridSize(1),VectorGridSize(2));

        P=getPressure(authkey, dataset, startTime, Lag6, ...
                      uCHIuInt,VectorGridSize(1)*VectorGridSize(2), ...
                      xpos(:,1:VectorGridSize(1)*VectorGridSize(2)));
        pdns(:,1:VectorGridSize(2),k)=reshape(P,VectorGridSize(1),VectorGridSize(2)); 

        Pgrad(:,1:VectorGridSize(1)*VectorGridSize(2)) = getPressureGradient(authkey, dataset, startTime, FD4Lag4, ...
                                                                             uCHIuInt,VectorGridSize(1)*VectorGridSize(2), ...
                                                                             xpos(:,1:VectorGridSize(1)*VectorGridSize(2)));
        dpdz(:,:,k)=reshape(Pgrad(3,:),VectorGridSize(1),VectorGridSize(2));
    end

save([sprintf(['DNS_V_A_P_t',num2str(startTime),'_5WU.mat'])],'udns','vdns','wdns','pdns','dudz', 'dpdz', '-v7.3');
end
            
