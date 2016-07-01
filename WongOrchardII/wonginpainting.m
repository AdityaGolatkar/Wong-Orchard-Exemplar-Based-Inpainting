% inpainting.m
 
% Inputs: 
%   - origImg        original image or corrupted 

%
% Outputs:
%   - inpaintedImg   The inpainted image; an MxNx3 matrix of doubles. 
%   - C              MxN matrix of confidence values accumulated over all iterations.
%   - D              MxN matrix of data term values accumulated over all iterations.


function [inpaintedImg,C,D] = wonginpainting(originalImage)
origImg = imread(originalImage);


fillColor = [0 255 0];
fillRegion = origImg(:,:,1)==fillColor(1) & origImg(:,:,2)==fillColor(2) & origImg(:,:,3)==fillColor(3);
psz = 5;
mask = fillRegion;

%% error check
if ~ismatrix(mask); error('Invalid mask'); end
if sum(sum(mask~=0 & mask~=1))>0; error('Invalid mask'); end
if mod(psz,2)==0; error('Patch size psz must be odd.'); end

%***************************
%*****Initialization********
%***************************

origImg = double(origImg);
%origImg = imgaussfilt(origImg);
img = origImg;
ind = img2ind(img);                                                           %ind is a M x N matrix with numbering as [1,m+1,2m+1.....;2.......;3........]
sz = [size(img,1) size(img,2)];
sourceRegion = ~fillRegion;

Co = sz(2) - psz + 1;
Ro = sz(1) - psz + 1;
m=1;
for PatchNo = 1:Ro*Co
  rowIndexes = rem((PatchNo -1),Ro) + 1 : rem((PatchNo -1),Ro) + psz ;
  colIndexes = (floor((PatchNo -1)/Ro) + 1: floor((PatchNo -1)/Ro) + psz )';
  XX = rowIndexes(ones(length(colIndexes),1),:);
  YY = colIndexes(:,ones(1,length(rowIndexes)));
  Patches(:,:,PatchNo) = (XX+(YY-1)*sz(1));                                   %Patches contains the indices of all the psz x psz patches in the image.

  Patchh = fillRegion(Patches(:,:,PatchNo));
  if any(Patchh(:))
      check(1,1,1,m) = 0;
  else
      check(1,1,1,m) = 1;
  end
  m=m+1;
  ImagePatchesPixels(:,:,:,PatchNo) = img(rowIndexes,colIndexes,:);
end 
%min(check)
%check(16446)
%[q,w] = min(check);
%check(16446:16546)


%*******************************
% **Initialize isophote values**
%*******************************
[Ix(:,:,3), Iy(:,:,3)] = gradient(img(:,:,3));
[Ix(:,:,2), Iy(:,:,2)] = gradient(img(:,:,2));
[Ix(:,:,1), Iy(:,:,1)] = gradient(img(:,:,1));
Ix = sum(Ix,3)/(3*255); Iy = sum(Iy,3)/(3*255);
temp = Ix; Ix = -Iy; Iy = temp;  % Rotate gradient 90 degrees


%****************************************
% *Initialize confidence and data terms**
%****************************************
C = double(sourceRegion);
D = repmat(-.1,sz);
iter = 1;

% Seed 'rand' for reproducible results (good for testing)
rand('state',0);


%*******************************************************
% ****Loop until entire fill region has been covered****
%*******************************************************
while any(fillRegion(:))
  
  % Find contour & normalized gradients of fill region
  fillRegionD = double(fillRegion); 
  dR = find(conv2(fillRegionD,[1,1,1;1,-8,1;1,1,1],'same')>0);                 %the numbering of the elements of matrices is in a vertical manner.


  %*****************************************************
  %**Computing the normal to the contour i.e.         **
  %**the boundary of the target and the source region.**
  %*****************************************************
  sRegion = double(~fillRegion);
  sRegion = imgaussfilt(sRegion);
  [Nx,Ny] = gradient(sRegion);                                     %gradient of the source region. 
  N = [Nx(dR(:)) Ny(dR(:))];                                                   %computing the X and Y components of the gradient on the boundary points.N will have 2 columns first for X and second for Y.
  N = normr(N);                                                                %we normalize the gradient to get the equivalent normal vector at each point on the boundary.
  N(~isfinite(N))=0; % handle NaN and Inf
  

  %*****************************************
  % Compute confidences along the fill front
  %*****************************************
  for k=dR'
    Hp = getpatch(sz,k,psz);
    q = Hp(~(fillRegion(Hp)));                                                %q corresponds to index values of 0 elements in the fillRegion matrix i.e the elements which are in the source region. 
    C(k) = sum(C(q))/numel(Hp);
  end
  
  %*******************************************************
  % Compute patch priorities = confidence term * data term
  %*******************************************************
  D(dR) = abs(Ix(dR).*N(:,1)+Iy(dR).*N(:,2)) + 0.001;
  priorities = C(dR).* D(dR);
  
  %*************************************
  % Find patch with maximum priority, Hp
  %*************************************
  [~,ndx] = max(priorities(:));
  p = dR(ndx(1));
  [Hp,rows,cols] = getpatch(sz,p,psz);
  toFill = fillRegion(Hp);                                                    %get a psz x psz patch filled with 1 and 0 depending on the Hp patch.
  

  %***************************************
  % Find exemplar that minimizes error, Hq
  %***************************************
  %function bestn = bestnmatches(n,img,img(rows,cols,:),toFill',Patches)
  [bestn modifiedssd] = bestnmatches(img,img(rows,cols,:),toFill',ImagePatchesPixels,check,Ro,Co);
  bestNpatchesNo = bestn(1:10);
  bestNssd = modifiedssd(bestNpatchesNo);
  %h =  mean(best5ssd);
  h=max(bestNssd);
  h=h*h*h*h;
  %size(h)
  weights = exp(-bestNssd/h);
  sumOFweights = sum(weights);
  %size(sumOFweights)
  %size(weights)
  %size(best10patchesNo)
  %size(best10ssd)
  %best10patchesNo is 10x1-------------best10ssd is 1x1x1x10
  
  %Hq = Patches(:,:,bestn);
  %Hq = bestexemplar(img,img(rows,cols,:),toFill',sourceRegion);
  
  %*******************
  % Update fill region
  %*******************
  toFill = logical(toFill);                   
  fillRegion(Hp(toFill)) = false;                                             %making all the ones to zeros in the fillRegion matrix.Implying that part has been filled.
  
  %***************************************
  % Propagate confidence & isophote values
  %***************************************
  C(Hp(toFill))  = C(p);
  %Ix(Hp(toFill)) = Ix(Hq(toFill));
  %Iy(Hp(toFill)) = Iy(Hq(toFill));
  

  %******************************
  % Copy image data from Hq to Hp
  %******************************

  for j=1:10
      Hq = Patches(:,:,bestNpatchesNo(j));
      ind(Hp(toFill)) = ind(Hq(toFill));                                          %Hp(toFill) gives the index of the elements which need to be filled. The elements lying in the target region.
      %img(rows,cols,:) = ind2img(ind(rows,cols),origImg);  
      patch(:,:,:,j) = weights(1,1,1,j)*ind2img(ind(rows,cols),origImg);
  end
    img(rows,cols,:) = sum(patch,4)/sumOFweights;

  iter = iter+1;
end

inpaintedImg = img;
imshow(uint8(inpaintedImg));
imwrite(uint8(inpaintedImg),'Result.png');

%*********************************************************
% Returns the indices for a pszxpsz patch centered at pixel p.
%*********************************************************
function [Hp,rows,cols] = getpatch(sz,p,psz)
% [x,y] = ind2sub(sz,p);  % 2*w+1 == the patch size
w=(psz-1)/2; p=p-1; y=floor(p/sz(1))+1; p=rem(p,sz(1)); x=floor(p)+1;                           %x is the row number and y is the column number
rows = max(x-w,1):min(x+w,sz(1));
cols = (max(y-w,1):min(y+w,sz(2)))';
Hp = sub2ndx(rows,cols,sz(1));

%************************************************************************
% Converts the (rows,cols) subscript-style indices to Matlab index-style
% indices.  Unforunately, 'sub2ind' cannot be used for this.
%************************************************************************
function N = sub2ndx(rows,cols,nTotalRows)
X = rows(ones(length(cols),1),:);
Y = cols(:,ones(1,length(rows)));
N = X+(Y-1)*nTotalRows;


%************************************************************************
% Converts an indexed image into an RGB image, using 'img' as a colormap
%************************************************************************
function img2 = ind2img(ind,img)
for i=3:-1:1, temp=img(:,:,i); img2(:,:,i)=temp(ind); end;


%***********************************************************************
% Converts an RGB image into a indexed image, using the image itself as
% the colormap.
%***********************************************************************
function ind = img2ind(img)
s=size(img); ind=reshape(1:s(1)*s(2),s(1),s(2));

%**********************************************
%Gives the pixel values of the best 10 patches
%**********************************************
%HpT is the transpose of the patch on the boundary.
function [bestn,modifiedssd] = bestnmatches(img,imgpatch,toFillT,ImagePatchesPixels,check,Ro,Co)
%size(imgpatch)
%size(ImagePatchesPixels)
diff = bsxfun(@minus,ImagePatchesPixels,imgpatch);
%diff = bsxfun(@times,diff,diff);
sqrdiff = diff.*diff;
sumsqrdiff = sum(sqrdiff,3);
sumsqrdiff = bsxfun(@times,sumsqrdiff,~toFillT);
ssd = sum(sumsqrdiff,1);
ssd = sum(ssd,2);
%min(ssd)
%min(check)
%size(check)
%R*C
%for i = 1:Ro*Co
%    modifiedssd(i) = ssd(1,1,1,i)*check(i);
%end
modifiedssd = ssd.*check;
%size(modifiedssd)
%min(modifiedssd)
%modifiedssd = ssd,check;
%nonzero = find(ssd>0);
nonzero = find(modifiedssd>0);
%[sorted sortInd] = sort(ssd(nonzero),'ascend');
[sorted sortInd] = sort(modifiedssd(nonzero));
%[temp,minPatchNo] = min(modifiedssd(nonzero));
%sorted(1)
%temp
%size(sortInd)
%nonzero(minPatchNo)
%nonzero(sortInd(1))
%x(1)
%bestn = nonzero(minPatchNo);
%bestn(1)
%minPatchNo
bestn = nonzero(sortInd);
