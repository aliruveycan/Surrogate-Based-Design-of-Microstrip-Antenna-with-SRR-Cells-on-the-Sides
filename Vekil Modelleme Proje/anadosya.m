%5G İçin yanlarında SRR hücreleri bulunan mikroşerit antenin
%elektromanyetik simülasyonlara dayalı veri üretimi aşaması

%1-CST'nin başlatılması
%kaynak kod yola eklenir.
addpath(genpath('CST-MATLAB-API-master'));
%cst uygulaması başlatılır.
cst = actxserver('CSTStudio.application');
%cst açıkken microwave studio başlatılabilir.
mws = cst.invoke('NewMWS');

%uzunluk birimi mikrometredir.(10^-6)
%Merkez frekansın %3'ü alt ve üst sınırlardır.
fcenter = 3.5;
fmin = fcenter * 0.97;
fmax = fcenter * 1.03;

CstDefineUnits(mws,'um','GHz','s','Kelvin','V','A','Ohm','S','PikoF','NanoH');
CstDefineFrequencyRange(mws,fmin,fmax);
CstMeshInitiator(mws);
%o zaman anten boş uzayda olacaktır.

%sınır koşulları
Xmin = 'expanded open';
Xmax = 'expanded open';
Ymin = 'expanded open';
Ymax = 'expanded open';
Zmin = 'expanded open';
Zmax = 'expanded open';

CstDefineOpenBoundary(mws,fmin,Xmin,Xmax,Ymin,Ymax,Zmin,Zmax);

%Ardından arka plan malzemesini tanımlıyoruz.
XminSpace = 0;
XmaxSpace = 0;
YminSpace = 0;
YmaxSpace = 0;
ZminSpace = 0;
ZmaxSpace = 0;

CstDefineBackroundMaterial(mws,XminSpace,XmaxSpace,YminSpace,YmaxSpace,ZminSpace,ZmaxSpace);

%Malzemelerin Tanımı

metal = CstCopperAnnealedLossy(mws);
[dielectric, epsilon_r] = CstFR4lossy(mws);
t = 0.035 * 1e3; %metal levhanın (bakır) yüksekliği

%Temel Antenin Tasarımı

%Veri kümesi oluşturulurken, metamalzemeler iyi parametrelere sahip bir
%temel antene uygulanacaktır.

h = 1700;

W  =0.9*3e8/(2*fcenter*1e9)*sqrt(2/(epsilon_r+1))*1e6; %radyasyon verimliliği
epsilon_reff = (epsilon_r+1)/2+((epsilon_r-1)/2).*(1+(12/W).*h).^(-1/2); %antenin etkin dielektrik sabiti
delta_l = 0.412.*h.*((epsilon_reff+0.3).*(W./h+0.264))./((epsilon_reff-0.258).*(W./h+0.8)); %yamanın gerçekte
%olduğundan ne kadar büyük göründüğü
l = 0.98.*3e5./(2*fcenter*sqrt(epsilon_reff))-2.*delta_l; 

Y0 = 4*h;
g = 1000;
W0 = 3.5 * g;

DrawMicrostripAntenna(mws,W,l,t,h,Y0,W0,g,metal,dielectric);
DrawPort(mws,W,l,t,h,W0);

%Temel Antenin Simülasyonu

base_path = 'C:\Kullanıcılar\aliru\results'; %sonuçların dışa aktarılacağı yol
CstDefineFarfieldMonitor(mws,strcat('farfield (f=', num2str(fcenter),')'),fcenter);
CstDefineTimedomainSolver(mws,-40);
ExportResults(mws,base_path,0,W,l,t,h,Y0,W0,g);
CstSaveProject(mws,strcat(base_path, '\0\simula.cst'));
CstQuitProject(mws);

%Metamalzeme hücresi için değer aralığı

lambda = 3e5./fcenter;

n_Wm = 5;

Wm_min = 0.025*lambda;
Wm_max = 1/4*lambda;
Wm_step = (Wm_max-Wm_min)/(n_Wm-1);
Wm = Wm_min:Wm_step:Wm_max;

tm = 0.1.*Wm;

n_W0m = 4;

W0m_min = 1.9e-3*lambda;
W0m_max = 7.6e-3*lambda;
W0m_step = (W0m_max-W0m_min)/(n_W0m-1);
W0m = W0m_min:W0m_step:W0m_max;

n_dm = 4;

dm_min = 0.9e-3*lambda;
dm_max = 5.7e-3*lambda;
dm_step = (dm_max-dm_min)/(n_dm-1);
dm = dm_min:dm_step:dm_max;

n_sim = n_Wm*n_W0m*n_dm;

%Pro-start değerlerinin aralığı

%Xa : düzenlemeden yamaya olan X cinsinden mesafe
%Ya : düzenlemenin hücreleri arasındaki Y cinsinden mesafe

rows = 3:2:7;

n_Xa = 2;

Xa = zeros(length(Wm),n_Xa);
for i = 1:length(Wm)
    Xa_min = 0;
    Xa_max = W/2-Wm(i)/2;
    Xa_step = (Xa_max-Xa_min)/(n_Xa-1);

    Xa(i,:) = Xa_min:Xa_step:Xa_max;
end
Xa = Flat(Xa);


n_Ya = 2;

Ya = zeros(length(Wm),n_Ya);
for i = 1:length(Wm)
    Ya_min = Wm(i);
    Ya_max = 4*l/(min(rows)-1)-Wm(i);
    Ya_step = (Ya_max-Ya_min)/(n_Ya-1);

    Ya(i,:) = Ya_min:Ya_step:Ya_max;
end
Ya = Flat(Ya);
Ya = Ya(Ya>=0);

%Simülasyon

%Hangi geometri kombinasyonlarının mümkün olduğunu kontrol etmek için 1:1
%kontrol yapmamız gerekir.

possible_geometry = [];
count = 1;
for wm = 1:length(Wm)
    for w0m = 1:length(W0m)
        for Dm = 1:length(dm)
            
            
            for row = 1:length(rows)
                for xa = 1:length(Xa)
                    if (xa <= W/2 - Wm(wm)/2) % dizilimin alt tabaka genişliğinin ötesine geçmediği anlamına gelir.
                        for ya = 1:length(Ya)
                            if(ya <= 4*l/(rows(row)-1) - Wm(wm)) % dizilimin alt tabaka yüksekliğinin ötesine geçmediği anlamına gelir.
                                possible_geometry(count, :) = [Wm(wm) tm(wm) W0m(w0m) dm(Dm) rows(row) Xa(xa) Ya(ya)];
                                count = count + 1;
                            end
                        end
                    end
                end
            end
        end
    end
end

fprintf(['Simulasyon sayisi: %d\n' ...
    'Tahmini tamamlanma suresi: %.0f gun'], count, 3*count/60/24);

%Olası düzenlemeleri oluşturmak için şunları yaparız:

count = 1;

for i = 1:length(possible_geometry)
    
    mws = cst.invoke('OpenFile', strcat(base_path, '\0\simula.cst'));
    
    CstDefineFrequencyRange(mws, fmin, fmax);
    CstMeshInitiator(mws);
    
    CstDefineOpenBoundary(mws, fmin, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax);
    
    CstDefineBackroundMaterial(mws, XminSpace, XmaxSpace, YminSpace, YmaxSpace, ZminSpace, ZmaxSpace);
    
    metal = CstCopperAnnealedLossy(mws);
    [dieletric, epsilon_r] = CstFR4lossy(mws);
    
    DrawPort(mws, W, l, t, h, W0);
    
    for side = [-1 1]
        for cell = 1:possible_geometry(i,5)
            
            Wm = possible_geometry(i,1);
            W0m = possible_geometry(i,3);
            dm = possible_geometry(i,4);
            tm = possible_geometry(i,2);
            hm = t+h;
            h0 = t;
            
            Xa = possible_geometry(i,6);
            Ya = possible_geometry(i,7);
            
            if cell == 1
                ya = 0;
            elseif mod(cell,2) == 0
                ya = (cell/2)*(Ya+Wm);
            else
                ya = (-1)*((cell-1)/2)*(Ya+Wm);
            end
            
            xa = (W/2+Xa+Wm/2)*side;
            
            DrawSquareSRR(mws,Wm,W0m,dm,tm,hm,h0,metal,cell*side,xa,ya)
        end
    end
    
    CstDefineFarfieldMonitor(mws,strcat('farfield (f=',num2str(fcenter),')'), fcenter);
   
    CstDefineTimedomainSolver(mws,-40);
    
    ExportResultsMeta(mws, 'C:\Kullanıcılar\aliru\results', count, Wm,W0m,dm,tm,hm,h0,cell,Xa,Ya);
    
    CstSaveAsProject(mws,strcat('C:\Kullanıcılar\aliru\results\',num2str(count),'\simula.cst'));
    CstQuitProject(mws);
    
    count = count + 1;
end

