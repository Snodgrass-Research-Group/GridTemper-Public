%% dump_reference.m — one-time MATLAB reference dump for GridTemper regression tests
%
% HOW TO RUN: open MATLAB and, in the Command Window, run these three lines
% (adjust the path to your clone of the MATLAB source repository):
%
%     cd '<path to TX-2030-Case-Updates>\Functions'
%     addpath('ConductorDataTables'); addpath('..\python\matlab_reference')
%     dump_reference
%
% IMPORTANT: invoke it as the command `dump_reference` (above), NOT via
% `run('...dump_reference.m')` — MATLAB's `run` temporarily cd's into the
% script's own folder, which breaks every relative path below (Data.mat,
% ConductorData69.mat, the ../python/tests/reference output dirs). Running it as
% a command keeps the current folder = Functions/, where those paths resolve.
% The two addpath lines put the script and the conductor .mat files on the path.
%
% It re-executes the main.m pipeline stage by stage, writing each intermediate
% result to ../python/tests/reference/ (CSVs under matlab_csv/, matrices as -v7
% .mat that scipy.io can read). Completes in well under a minute.
%
% After it finishes, re-run the data converter to pick up the XF tables:
%     cd ../python && python -m uv run python scripts/convert_mat_data.py
%
% NOTE: main.m as committed would ERROR at ACPowerFlowFinal line 100 because
% codectrl.RunDebuggingCode is never set (a known blocker in the
% MATLAB code). This script sets it explicitly.

%% Output folders
outdir = fullfile('..','python','tests','reference');
csvdir = fullfile(outdir,'matlab_csv');
if ~exist(outdir,'dir'), mkdir(outdir); end
if ~exist(csvdir,'dir'), mkdir(csvdir); end

%% XF tables (MATLAB table objects inside Data.mat — scipy cannot read these)
load('Data.mat','Data')
writetable(Data.XFData, fullfile(csvdir,'xf_data.csv'));
writetable(Data.XFXR,   fullfile(csvdir,'xf_xr.csv'));

%% ===== main.m, stage by stage, with dumps ==============================
region = {'SPP'};
codectrl = struct;
codectrl.RunDebuggingCode = 0;   % see header note / blocker B4
codectrl.SaveResults = 0;
PF.SNetFull = 1;
options = struct;
options.Region = region;
options = InitOptions( options );

% ---- Stage: load_inputs (convertPWtoTable) ----
filename_gen = "Input data from PW/GenBuses.csv";
filename_gencost = "Input data from PW/GenCostCurves.csv";
filename_bus = "Input data from PW/Buses.csv";
filename_branch = "Input data from PW/Branches.csv";
GenInfo = convertPWtoTable(filename_gen);
BusInfo = convertPWtoTable(filename_bus);
BranchInfo = convertPWtoTable(filename_branch);
nrow = size(BranchInfo,1);
BranchInfo.Tap = zeros(nrow, 1);
BranchInfo.Resize = zeros(nrow, 1);
GenCostInfo = convertPWtoTable(filename_gencost);
writetable(GenInfo,     fullfile(csvdir,'stage1_gen.csv'));
writetable(BusInfo,     fullfile(csvdir,'stage1_bus.csv'));
writetable(BranchInfo,  fullfile(csvdir,'stage1_branch.csv'));
writetable(GenCostInfo, fullfile(csvdir,'stage1_gencost.csv'));

% ---- Stage: split_gsus ----
[ModifiedBusInfo,ModifiedBranchInfo,ModifiedGenInfo,ModifiedGencostInfo] = split_add_GSUs(GenInfo,BusInfo,BranchInfo,GenCostInfo);
writetable(ModifiedBusInfo,     fullfile(csvdir,'stage2_bus.csv'));
writetable(ModifiedBranchInfo,  fullfile(csvdir,'stage2_branch.csv'));
writetable(ModifiedGenInfo,     fullfile(csvdir,'stage2_gen.csv'));
writetable(ModifiedGencostInfo, fullfile(csvdir,'stage2_gencost.csv'));

% ---- Stage: fit_costs ----
[ModifiedGenCost_quadratic] = costcurve_to_cubic(ModifiedGencostInfo);
writetable(ModifiedGenCost_quadratic, fullfile(csvdir,'stage3_gencost_quadratic.csv'));

% ---- Stage: demand + renumber (main.m lines 30-59, verbatim) ----
filename_demand = "Input data from PW/Demand_onetimepoint.csv";
DemandInfo = convertPWtoTable(filename_demand);

ModifiedBusInfo.NumExt = ModifiedBusInfo.Number;
ModifiedBusInfo.NumInt = (1:height(ModifiedBusInfo))';
ModifiedBusInfo.Number = (1:height(ModifiedBusInfo))';
ModifiedBranchInfo.FromExt = ModifiedBranchInfo.FromNumber;
ModifiedBranchInfo.ToExt = ModifiedBranchInfo.ToNumber;
ModifiedGenInfo.BusNumExt = ModifiedGenInfo.NumberOfBus;
ModifiedGenCost_quadratic.BusNumExt = ModifiedGenCost_quadratic.NumberOfBus;

buses = height(ModifiedBusInfo);
for row = 1:buses
    FromLines = ModifiedBranchInfo.FromExt==ModifiedBusInfo.NumExt(row);
    ToLines = ModifiedBranchInfo.ToExt==ModifiedBusInfo.NumExt(row);
    ModifiedBranchInfo.FromInt(FromLines) = row;
    ModifiedBranchInfo.ToInt(ToLines) = row;
    GenIndex = ModifiedGenInfo.NumberOfBus == ModifiedBusInfo.NumExt(row);
    ModifiedGenInfo.NumberOfBus(GenIndex) = row;
    ModifiedGenCost_quadratic.NumberOfBus(GenIndex) = row;
    LoadIndex = DemandInfo.BusNumber == ModifiedBusInfo.NumExt(row);
    DemandInfo.BusNumber(LoadIndex) = row;
end
writetable(ModifiedBusInfo,           fullfile(csvdir,'stage4_bus.csv'));
writetable(ModifiedBranchInfo,        fullfile(csvdir,'stage4_branch.csv'));
writetable(ModifiedGenInfo,           fullfile(csvdir,'stage4_gen.csv'));
writetable(ModifiedGenCost_quadratic, fullfile(csvdir,'stage4_gencost.csv'));
writetable(DemandInfo,                fullfile(csvdir,'stage4_demand.csv'));

% ---- Stage: build_snet (Create_SNet -> UCED) ----
[SNet] = Create_SNet(DemandInfo,ModifiedGenCost_quadratic,options,buses);
save(fullfile(outdir,'stage5_snet.mat'),'SNet','-v7');

% ---- Stage: build_incidence ----
params.NumBuses = buses;
FromNet = ModifiedBranchInfo.FromInt;
ToNet = ModifiedBranchInfo.ToInt;
[ A, AFull ] = BuildANet(FromNet,ToNet,params.NumBuses);
save(fullfile(outdir,'stage6_incidence.mat'),'A','-v7');

% ---- Stage: assign_conductors ----
ConductorData = CreateConductorData(options);
writetable(ConductorData, fullfile(csvdir,'stage7_conductor_data.csv'));
ModifiedBranchInfo = AssignConductors(ModifiedBranchInfo, ConductorData, options);
writetable(ModifiedBranchInfo, fullfile(csvdir,'stage7_branch.csv'));

% ---- Ybus variants straight from YBusCalc (powerflow-core regression) ----
[YBusShort,YBusLong,YPrim,BMatrix,~,BBusShort,BBusLong,~] = YBusCalc( A,FromNet,ToNet,ModifiedBranchInfo.R,ModifiedBranchInfo.X,ModifiedBranchInfo.B,0,0 );
save(fullfile(outdir,'ybus_calc.mat'),'YBusShort','YBusLong','YPrim','BMatrix','BBusShort','BBusLong','-v7');

% ---- DC power flow reference (dormant path, deterministic) ----
[ BBusDC,~,~] = YBusCalcDC( A,ModifiedBranchInfo.X,0 );
[~,DCPFlows] = DCPowerFlowSimple( 1, sparse(ModifiedBranchInfo.X), (real(SNet)/options.SBase), A,[],0 );
save(fullfile(outdir,'dc_reference.mat'),'BBusDC','DCPFlows','-v7');

% ---- Stage: reactive_placement (the full active path) ----
load('QNCAResults.mat', 'params')
params.TotalPaths = height(ModifiedBranchInfo);
params.Scenarios = 1;
[ACPF,ModifiedBranchInfo,ModifiedBusInfo] = ReactivePowerPlacement(ModifiedBranchInfo,ModifiedBusInfo,ModifiedGenInfo,PF,params,options,codectrl,SNet);
writetable(ModifiedBranchInfo, fullfile(csvdir,'stage8_branch.csv'));
writetable(ModifiedBusInfo,    fullfile(csvdir,'stage8_bus.csv'));
% ACPF struct: keep the numeric fields scipy can read (drop tables/cells).
% Which fields ACPowerFlowFinal fills depends on whether the detailed solve was
% flagged singular (it only sets VBus/VBusPolar/PLine*/SNetCalc/Converge inside
% `if ~PF.SingularY`). Guard every field and print diagnostics so a singular
% base case is obvious in the log rather than crashing the dump.
if isfield(ACPF,'SingularY'); fprintf('DIAG ACPF.SingularY = %g\n', ACPF.SingularY); end
if isfield(ACPF,'Converge'); fprintf('DIAG ACPF.Converge = %g\n', ACPF.Converge); else; disp('DIAG ACPF.Converge NOT set (detailed solve skipped -> singular)'); end
disp('DIAG ACPF fields:'); disp(fieldnames(ACPF)');
acpf_dump = struct;
numeric_fields = {'VBus','VBusPolar','PLineSend','PLineRec','PLineFinal', ...
    'SNetCalc','Converge','SingularY','GenList','LoadList','VBusSet', ...
    'dVdQ','BusQInjection','AngleDifference'};
for k = 1:numel(numeric_fields)
    fld = numeric_fields{k};
    if isfield(ACPF,fld); acpf_dump.(fld) = ACPF.(fld); end
end
if isfield(ACPF,'Stats')
    acpf_dump.Losses = ACPF.Stats.Losses(1);
    acpf_dump.LossPercent = ACPF.Stats.LossPercent(1);
end
save(fullfile(outdir,'stage8_acpf.mat'),'-struct','acpf_dump','-v7');

disp('dump_reference.m complete. Now re-run: python -m uv run python scripts/convert_mat_data.py')
