clear; clc; close all;

%% ===================== 0. File path and parameters =====================

dataDir = 'C:\Users\90743\Documents\MATLAB\Project_Ast';
infile  = fullfile(dataDir, 'result.txt');

% Coma cluster center
RA0  = 194.935;   % deg
DEC0 = 27.935;    % deg

% Coma systemic redshift
zc = 0.0231;

% Speed of light
c = 299792.458;   % km/s

% Task 1 criteria
Rmax = 1.5;       % deg
zmin = 0.017;
zmax = 0.028;

% Task 2 sigma-clipping criteria
seedWindow_kms = 3000;   % initial velocity window
clipSigma = 3;
maxIter = 20;

%% ===================== 1. Read NED result.txt =====================

raw = readlines(infile);
raw = string(raw);

% Locate table header
headerIdx = find(startsWith(strtrim(raw), ...
    "No.|Object Name|RA|DEC|Type|Velocity|Redshift"), 1);

if isempty(headerIdx)
    error('Cannot find NED table header in result.txt.');
end

headerLine = strtrim(raw(headerIdx));
dataLines = raw(headerIdx+1:end);

% Keep only real table rows: rows beginning with number + |
isTableRow = false(size(dataLines));
for i = 1:numel(dataLines)
    isTableRow(i) = ~isempty(regexp(char(dataLines(i)), '^\s*\d+\|', 'once'));
end
dataLines = dataLines(isTableRow);

% Split by |
headers = strtrim(split(headerLine, "|"))';
varNames = matlab.lang.makeValidName(headers);
varNames = matlab.lang.makeUniqueStrings(varNames);

parts = split(dataLines, "|");
parts = strtrim(parts);

T = array2table(parts, 'VariableNames', varNames);

% Convert numeric columns
numVars = {'No_', 'RA', 'DEC', 'Velocity', 'Redshift', 'Separation', ...
           'References', 'Notes', 'PhotometryPoints', 'Positions', ...
           'RedshiftPoints', 'DiameterPoints', 'Associations'};

for i = 1:numel(numVars)
    vname = numVars{i};
    if ismember(vname, T.Properties.VariableNames)
        T.(vname) = str2double(T.(vname));
    end
end

% String columns
if ismember('ObjectName', T.Properties.VariableNames)
    T.ObjectName = string(strtrim(T.ObjectName));
end

if ismember('Type', T.Properties.VariableNames)
    T.Type = string(strtrim(T.Type));
end

%% ===================== 2. Basic variables =====================

ra  = T.RA;
dec = T.DEC;
z   = T.Redshift;
typ = string(T.Type);

% Use strict galaxy type: Type = G
valid = ~isnan(ra) & ~isnan(dec) & ~isnan(z);
isGalaxy = valid & strcmpi(strtrim(typ), "G");

%% ===================== 3. Angular distance and relative velocity =====================

% Angular distance from Coma center
raRad   = deg2rad(ra);
decRad  = deg2rad(dec);
ra0Rad  = deg2rad(RA0);
dec0Rad = deg2rad(DEC0);

cosang = sin(dec0Rad).*sin(decRad) + ...
         cos(dec0Rad).*cos(decRad).*cos(raRad - ra0Rad);

cosang = min(max(cosang, -1), 1);
rDeg = rad2deg(acos(cosang));

% Relative line-of-sight velocity
vRel = c .* (z - zc) ./ (1 + zc);

%% ===================== 4. Task 1: redshift-window selection =====================

isSpatialSample = isGalaxy & rDeg <= Rmax;

isTask1Member = isSpatialSample & ...
                z >= zmin & z <= zmax;

fprintf('\n===== Task 1: redshift-window selection =====\n');
fprintf('Spatial sample galaxies, R <= %.2f deg: %d\n', Rmax, sum(isSpatialSample));
fprintf('Task 1 redshift-window members: %d\n', sum(isTask1Member));
fprintf('Mean z of Task 1 members: %.6f\n', mean(z(isTask1Member), 'omitnan'));
fprintf('Std z of Task 1 members : %.6f\n', std(z(isTask1Member), 'omitnan'));
fprintf('Mean v_rel of Task 1 members: %.2f km/s\n', mean(vRel(isTask1Member), 'omitnan'));
fprintf('Std v_rel of Task 1 members : %.2f km/s\n', std(vRel(isTask1Member), 'omitnan'));

%% ===================== 5. Task 2: 3-sigma clipping =====================

% Initial seed for sigma clipping
isSigmaMember = isSpatialSample & abs(vRel) <= seedWindow_kms;

fprintf('\n===== Task 2: 3-sigma clipping =====\n');
fprintf('Initial seed: |v_rel| <= %.0f km/s\n', seedWindow_kms);

for iter = 1:maxIter

    oldMask = isSigmaMember;

    vNow = vRel(isSigmaMember);
    vNow = vNow(~isnan(vNow));

    muNow = mean(vNow, 'omitnan');
    sigNow = std(vNow, 'omitnan');

    lowerNow = muNow - clipSigma * sigNow;
    upperNow = muNow + clipSigma * sigNow;

    isSigmaMember = isSpatialSample & ...
                    vRel >= lowerNow & vRel <= upperNow;

    fprintf('Iter %2d: N = %4d, mean = %8.2f km/s, sigma = %8.2f km/s, limits = [%8.2f, %8.2f]\n', ...
        iter, sum(isSigmaMember), muNow, sigNow, lowerNow, upperNow);

    if isequal(oldMask, isSigmaMember)
        fprintf('Sigma-clipping converged at iteration %d.\n', iter);
        break;
    end
end

vSigma = vRel(isSigmaMember);
muSigma = mean(vSigma, 'omitnan');
sigSigma = std(vSigma, 'omitnan');

finalLower = muSigma - clipSigma * sigSigma;
finalUpper = muSigma + clipSigma * sigSigma;

fprintf('\nFinal Sigma-clipping members: %d\n', sum(isSigmaMember));
fprintf('Final mean v_rel: %.2f km/s\n', muSigma);
fprintf('Final sigma v_rel: %.2f km/s\n', sigSigma);
fprintf('Final 3-sigma limits: [%.2f, %.2f] km/s\n', finalLower, finalUpper);

%% ===================== 6. Method comparison numbers =====================

isOverlap   = isTask1Member & isSigmaMember;
isTask1Only = isTask1Member & ~isSigmaMember;
isSigmaOnly = isSigmaMember & ~isTask1Member;

N_task1     = sum(isTask1Member);
N_sigma     = sum(isSigmaMember);
N_overlap   = sum(isOverlap);
N_task1only = sum(isTask1Only);
N_sigmaonly = sum(isSigmaOnly);

fprintf('\n===== Numbers for report table =====\n');
fprintf('Task 1 redshift-window members : %d\n', N_task1);
fprintf('3-sigma clipping members       : %d\n', N_sigma);
fprintf('Overlap members                : %d\n', N_overlap);
fprintf('Task 1 only                    : %d\n', N_task1only);
fprintf('3-sigma clipping only          : %d\n', N_sigmaonly);

%% ===================== 7. Figure 1: RA-Dec spatial distribution =====================

figure('Color', 'w', 'Position', [100 100 850 700]);

scatter(ra(isGalaxy & ~isTask1Member), dec(isGalaxy & ~isTask1Member), ...
    10, [0.72 0.72 0.72], 'filled');
hold on;

scatter(ra(isTask1Member), dec(isTask1Member), ...
    16, 'r', 'filled');

plot(RA0, DEC0, 'kp', 'MarkerSize', 15, ...
    'MarkerFaceColor', 'y', 'LineWidth', 1.2);

theta = linspace(0, 2*pi, 400);
circleDec = DEC0 + Rmax * sin(theta);
circleRA  = RA0  + Rmax * cos(theta) ./ cosd(DEC0);
plot(circleRA, circleDec, 'k--', 'LineWidth', 1.3);

set(gca, 'XDir', 'reverse');
xlabel('RA (deg)');
ylabel('Dec (deg)');
title('Spatial distribution of Coma candidate and selected member galaxies');

legend('Non-member galaxies in broad sample', ...
       'Selected member galaxies', ...
       'Coma center', ...
       'R = 1.5 deg selection radius', ...
       'Location', 'southwest');

grid on;
box on;

%% ===================== 8. Figure 2: radius-redshift selection diagram =====================

figure('Color', 'w', 'Position', [130 130 900 650]);

scatter(rDeg(isGalaxy & ~isTask1Member), z(isGalaxy & ~isTask1Member), ...
    10, [0.72 0.72 0.72], 'filled');
hold on;

scatter(rDeg(isTask1Member), z(isTask1Member), ...
    16, 'r', 'filled');

xline(Rmax, 'k--', 'LineWidth', 1.5);
yline(zmin, 'r--', 'LineWidth', 1.3);
yline(zc,   'k-',  'LineWidth', 1.5);
yline(zmax, 'r--', 'LineWidth', 1.3);

xlabel('Angular distance from Coma center (deg)');
ylabel('Redshift z');
title('Membership selection in radius-redshift space');

legend('Non-member galaxies in broad sample', ...
       'Selected member galaxies', ...
       'Radius limit', ...
       'Redshift limits', ...
       'Cluster systemic redshift', ...
       'Location', 'northeast');

xlim([0 2.5]);
ylim([0.005 0.060]);
grid on;
box on;

%% ===================== 9. Figure 3: sigma-clipping velocity histogram =====================

figure('Color', 'w', 'Position', [160 160 850 600]);

vSpatial = vRel(isSpatialSample);
vSpatial = vSpatial(~isnan(vSpatial));

vSig = vRel(isSigmaMember);
vSig = vSig(~isnan(vSig));

histogram(vSpatial, 'BinWidth', 200, ...
    'FaceColor', [0.82 0.82 0.82], ...
    'EdgeColor', 'none');
hold on;

histogram(vSig, 'BinWidth', 200, ...
    'FaceColor', [1.00 0.35 0.35], ...
    'EdgeColor', 'none');

xline(0, 'k-', 'LineWidth', 1.6);
xline(finalLower, 'r--', 'LineWidth', 1.5);
xline(finalUpper, 'r--', 'LineWidth', 1.5);

xlabel('Line-of-sight velocity relative to Coma center (km/s)');
ylabel('Number of galaxies');
title('Task 2: 3-sigma clipping in velocity space');

legend('Spatial sample', ...
       'Final Sigma-clipping members', ...
       'Coma systemic velocity', ...
       'Final 3\sigma limits', ...
       'Location', 'northeast');

grid on;
box on;