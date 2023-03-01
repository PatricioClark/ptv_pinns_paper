%% Perform SVDInterpolation To Obtain Velocity and Acceleration on Desired Grid Points
%% Jin Wang
%% Date 10.24.2017
% clear clc;
authkey = 'edu.jhu.pha.turbulence.testing-201311';
dataset = 'channel';

% Addpath
addpath('./turbmat/')

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
delta=2.5;
deltanu=1.0006e-3;
delta = delta*deltanu*1000;
delta = 2.5015;

Xg = -52.63547:delta:42.42155;
Yg = 45.34567:-delta:-49.71135;
Zg = -21.3455:delta:21.1801;
% Xg = Xg/1000.0;
% Yg = Yg/1000.0;
% Zg = Zg/1000.0;
Xg = Xg*deltanu;
Yg = Yg*deltanu;
Zg = Zg*deltanu;
N1=size(Xg,2);
N2=size(Yg,2);
N3=size(Zg,2);
totalN=N1*N2*N3;
N=totalN;
ut=4.9968e-2;
dt=0.0065/2;

% for startTime=9.0025:dt:9.0025+2000*dt
for startTime=10.00675:dt:9.0025+2000*dt
    % if startTime<10.31876
    if startTime<14.2611
        continue
    end
    nu=5*10^-5;
    root='D:\SyntheticPIV\x64\Debug';

    VectorGridSize=[N1,N2,N3];

    xshift=53.7823/1000.;
    yshift=51.0883/1000.;
    zshift=35.0/1000.;
    xshift=53.7823*deltanu;
    yshift=51.0883*deltanu;
    zshift=35.0*deltanu;

    y0=-1;
    x0=4*pi;
    z0=20*deltanu;
    % x0=0.0;
    % y0=0.0;
    % z0=0.0;

    x0 = x0 + xshift;
    y0 = y0 + yshift;
    z0 = z0 + zshift;

    udns=zeros(VectorGridSize(1),VectorGridSize(2),VectorGridSize(3));
    vdns=zeros(VectorGridSize(1),VectorGridSize(2),VectorGridSize(3));
    wdns=zeros(VectorGridSize(1),VectorGridSize(2),VectorGridSize(3));
    lamb=zeros(VectorGridSize(1),VectorGridSize(2),VectorGridSize(3));
    pdns=zeros(VectorGridSize(1),VectorGridSize(2),VectorGridSize(3));
    Pgrad=zeros(3,VectorGridSize(1)*VectorGridSize(2));

    for k=1:VectorGridSize(3)
        for j=1:VectorGridSize(2)
            for i=1:VectorGridSize(1)
                xpos(1,i+(j-1)*VectorGridSize(1))=Xg(i)+x0;
                xpos(2,i+(j-1)*VectorGridSize(1))=Yg(j)+y0;
                xpos(3,i+(j-1)*VectorGridSize(1))=Zg(k)+z0;
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

    startTime
    save([sprintf(['DNS_V_A_P_t',num2str(startTime, '%.5f'),'_5WU.mat'])],'udns','vdns','wdns','pdns','dudz', 'dpdz', '-v7.3');
end
            
