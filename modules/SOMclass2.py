from scipy.io import loadmat
import pandas as pd
import cv2
import numpy as np
import scipy.ndimage
import colorsys
import matplotlib.pyplot as plt
import os
import os.path

#### SOM Class ####
#  contains methods:
#
### Initialization Functions
#  - readtrainedSOM 
#  - readPCAcoef
#  - loadcellclasses
#
### Fetching data structures  
#  - getsortedcelldict
#  - getsortedclassdict
#  - getsorteddataframe
#
### Tools
#  - readimgandfilter
#  - segimg
#
### Creating Figures
#  - printcellfig
#  - printnodefig
#  - 
####

class SOM:
    def __init__(self):
        # Contains a list of all paths that have been segmented
        self.pathlist = []
        
        # Size of the cropped image of cells
        self.maxsize = 170
        
        # Connectivity to use when finding connected components
        self.connectivity = 4

        # Color Permutation Matrix     
        self.color_permute = []  
 
        # Contains the user-specified cell classes, read in using method
        #        'loadcellclasses'
        self.cellclasses = []
        
        # created in 'segimg', a list of with length equal the number of cells
        # showing the correct node for each cell
        self.correctnodedict = {}
        
        # created in 'segimg', contains array of cell images
        self.cellimagedict = {}
        
        # See method 'getsortedcelldict'
        self.sortedcelldict = {}
        
        # See method 'getsortedclassdict'
        self.sortedclassdict = {}
        
        # Number of features used, currently only one - number of cells
        #         in each SOM node
        self.numfeatures = 1 
      
        # a list containing the extracted features for each image
        self.sorteddataframe = []    

    def readtrainedSOM(self, path):
        # Input:
        # - path - path of the trained SOM vectors
        self.SOMvecs = loadmat(path)['trainedvecs7x7'].T #(1000,49)
        
    def readPCAcoef(self, path):
        # Input:
        # - path - path of the trained PCA vectors, used to reduce 
        #         each image to a 1000 dimensional vector
        self.PCAcoef = loadmat(path)['vectors']    #(imgsize*imgsize, 1000)
 
    def loadcellclasses(self, cellclasses):
        # Input:
        # - cellclasses - a list of lists containing the node ID's that 
        #         are grouped into each class
        self.cellclasses = cellclasses
        self.color_permute = list(range(len(self.cellclasses)))

    def loadcolorpermute(self, colorpermute):
        # Input:
        # - colorpermute - a list containing numbers indicating how colors will be permuted   
        # 
        self.color_permute = colorpermute
 
    def getsortedcelldict(self, path):
        # Input:
        # - path - path of the image for which we want the sortedcelldict
        # Returns:
        # - sortedcelldict - a dictionary organized by the index of the cell,
        #         containing images linked with class, node id
        #         key - index of cell - celldict['10']
        #         value - list - [class, node id, img] 
        return (self.sortedcelldict[path])

    def getsortedclassdict(self, path):
        # Input:
        # - path - path of the image for which we want the sortedclassdict
        # Returns:
        # - sortedcelldict - a dictionary organized by the index of the class,
        #         containing images linked with class, node id
        #         key - index of cell - celldict['10']
        #         value - list - [node id, cellid, img] 
        return (self.sortedclassdict[path])
     
    def getsorteddataframe(self):
        # Input:
        # - path - path of the image for which we want the sortedclassdict
        # Returns:
        # - sorteddataframe - a list containing all the features for each image
        #         with shape [number of images, number of features]
        return (np.concatenate(self.sorteddataframe, axis = 0))
        #return ((self.sorteddataframe))

    def readimgandfilter(self, path):
        # This function reads in the image and filters out cells with bad shapes
        # or with areas that are too small or too large
        #
        # Input:
        # - path - path of the image to process
        # Returns:
        # - stats - statistics for connected components of image, 3rd element of output
        #        from cv2.connectedComponentsWithStats()
        # - cells - image of segmented cells, with inside of cells = 1
        
        img = cv2.imread(path,0)
        assert len(np.unique(img)) == 2, "Input image should be a segmented cell image with only 2 values."
        
        # set mask to only 0's and 1's
        img_min = np.min(img); img_max = np.max(img);
        img[img == img_min] = 0
        img[img == img_max] = 1
        
        # cells at segmented high level
        cells = np.uint8(img==1)
        
        connectivity = self.connectivity 
        ccs = cv2.connectedComponentsWithStats(cells,connectivity , cv2.CV_32S)
        stats = pd.DataFrame(ccs[2]) #2389,5

        # Sort by area occupied, pix is 0.2 um, say smallest cell object is 1x1 um
        maxarea = 200*200
        minarea = 5*5
        maxlength = 150 #largest allowable cell object has dimension 30 um = 30/0.2 = 150 pixels
        
        # Filtering of bad area or bad shape
        # Remove cells that have too small or too large area
        # as well as cells that are too thin or too long
        badarea = (stats.loc[:,4] > maxarea) | (stats.loc[:,4] < minarea)
        badshape = (stats.loc[:,2] < 4) | (stats.loc[:,3] < 4) | (stats.loc[:,2] > maxlength) | (stats.loc[:,3] > maxlength)
        
        stats = stats.drop(stats[badarea | badshape].index)
        stats = stats.reset_index(drop=True)

        return(stats, cells)

    def printcellfig(self, path):  
        # This function takes a stored image path and generates
        # colors each cell in the image based on its SOM class
        #
        # Input:
        # - path - path of the image to process
        # Returns:
        # - newimg - image with cells colored based on SOM class
        ###########
        
        stats, cells = self.readimgandfilter(path)
        celldict = self.getsortedcelldict(path)
        numcellclasses = len(self.cellclasses)
        print('length of cell classes: ', len(self.cellclasses))
        print('length of cell dict: ', len(celldict.keys()))
        #print(celldict.keys())
        print('shape of stats: ',stats.shape)
 
        newimg= np.zeros((cells.shape[0],cells.shape[1],3))
        for i, rows in stats.iterrows():
            # If cell was filtered out, just continue
            if str(i) not in celldict:
                continue
                
            # Obtain the cropped images
            cropx = np.arange(stats.loc[i,1],stats.loc[i,1]+stats.loc[i,3]-1)
            cropy = np.arange(stats.loc[i,0],stats.loc[i,0]+stats.loc[i,2]-1)
            cropimg = cells[stats.loc[i,1]:stats.loc[i,1]+stats.loc[i,3]-1,stats.loc[i,0]:stats.loc[i,0]+stats.loc[i,2]-1]  
            x,y = np.where(cropimg == 1)
             
            # Set color for each cell based on the designated class for the cell
            newcrop = np.zeros((cropimg.shape[0], cropimg.shape[1], 3))
            rgb = colorsys.hsv_to_rgb(self.color_permute[celldict[str(i)][0]]/numcellclasses, 1., 1.) 
            newcrop[x,y,:] = rgb           
 
            # Place cell back in image
            newimg[stats.loc[i,1]:stats.loc[i,1]+stats.loc[i,3]-1,stats.loc[i,0]:stats.loc[i,0]+stats.loc[i,2]-1,:] = newcrop 
    
        plt.figure(figsize = (20,20))
        plt.tick_params(axis='both', labelsize = 32)
        plt.imshow(newimg)
        
        return(newimg)
 
    def segimg(self, path):
        # Core function that extracts all segmented cells in an image, rotates them, 
        # and finds the correct SOM class for each cell
        #
        # Input:
        # - path - path of the image to process
        #####
        
        # Append path to internal self.pathlist
        self.pathlist.append(path)
        connectivity = self.connectivity

        #stats is dataframe containing all sorted and filtered cells
        stats, cells = self.readimgandfilter(path)

        ##   Inside the Loop We: Rotate all images to align horizontally, 
        ##   create cellmat, a matrix containing images (vectors of length 'self.maxsize' ^2 ) of all cells 
        ##
        
        cellmat =[]
        import collections
        celldict_temp = collections.OrderedDict() 

        # for every segmented cell in the image: 
        for i, row in stats.iterrows():
            cropimg = cells[stats.loc[i,1]:stats.loc[i,1]+stats.loc[i,3]-1,stats.loc[i,0]:stats.loc[i,0]+stats.loc[i,2]-1]
           
            # stats2 used to find largest connected component in crop, crop can sometimes contain
            # more than one cell-like object.  I believe stats2's functionality can be merged into 
            # the method self.readimgandfilter
            _, output2, stats2, _  = cv2.connectedComponentsWithStats(cropimg,connectivity, cv2.CV_32S)
            maxind = np.argmax(stats2[1:,-1])+1  #stats[:,-1] is list of areas
            cropimg_ = np.zeros(cropimg.shape)
            cropimg_[output2 == maxind] = 1

            # get nonzero x and y coordinates
            nonzero = np.nonzero(cropimg_) 

            # Get image as 2 lists of coordinates.  Perform PCA on each cell to rotate cells all horiztonally
            imagevec = np.vstack(list(nonzero))
            covariance = np.cov(imagevec)
            w,v = np.linalg.eig(covariance)
            v_1 = v[:,np.argmax(w)]

            if v_1[0] == 0:
                angle = 0
            else:
                #angle =  -np.arctan(v_1[1]/v_1[0])*180/np.pi-90;
                angle =  -np.arctan(v_1[1]/v_1[0])*180/np.pi;
   
            rotated = np.uint8(np.round(scipy.ndimage.rotate(cropimg_,angle,order=5)))
    
            # stats3 is used to recrop the image, as after rotation, the crop area is often increased
            _, _, stats3, _  = cv2.connectedComponentsWithStats(rotated,connectivity, cv2.CV_32S)
    
            if (stats3.shape[0] == 1): #implies stats3 is empty
                #print('skipped')
                continue
        
            cropped = rotated[stats3[1,1]:stats3[1,1]+stats3[1,3]-1,stats3[1,0]:stats3[1,0]+stats3[1,2]-1]
            maxsize = self.maxsize  # maximum size of image
            
            newimage = np.zeros((maxsize,maxsize))
            nb = np.array(newimage.shape)
            na = np.array(cropped.shape)
            start = (nb - na) // 2
            
                # If cropped image of cell begins outside of image, omit
            if start[0] < 0 or start[1] < 0:
                continue

            newimage[start[0]:start[0]+na[0], start[1]:start[1]+na[1]] = cropped

            # cellmat is an array of images
            cellmat.append(np.reshape(newimage,(-1,1)))# change picture to one column of values describing the picture
            
            # celldict_temp is the same things as cellmat, but with images resized to (maxsize,maxsize)
            celldict_temp[str(i)] = np.reshape(newimage,(self.maxsize,self.maxsize))#this is what you want to count the statistics on

        # Add array of cell images to internal variable 'self.cellimagedict'
        cellmat = np.concatenate(cellmat,axis=1) #(imgsize*imgsize, 3435) numpy array of 3 dimentions, x,y,cell number
        plt.plot(cellmat[:,0])
        self.cellimagedict[path] = cellmat

        # Retrieve some parameters
        numclasses = self.SOMvecs.shape[1]
        numcells = cellmat.shape[1]

        # Reduce dimensionality of cell images in cellmat from (maxsize * maxsize) to 1000
        # using PCA components
        cellmat_r = np.dot(self.PCAcoef.T, cellmat)
        cells3d = np.tile(cellmat_r,(numclasses,1,1)) #49,1000,3435
        classes3d = np.transpose(np.tile(self.SOMvecs,(numcells,1,1)), (2,1,0))#49,1000,3435
        classmatrix = (np.sum((cells3d - classes3d)**2, axis=1)) #49,3435
        correctnode = np.argmin(classmatrix,axis=0) #3435
        self.correctnodedict[path] = correctnode

        # Sort cells into correct class, create nodedict, which will be useful below
        nodedict = {}
        for k,cell_class in enumerate(self.cellclasses):
            for node in cell_class:
                nodedict[str(node)] = k
        
        # self.sorteddataframe (not really a dataframe)
        features_list = self.sorteddataframe
        features_new = np.zeros((1,len(self.cellclasses),self.numfeatures))

        # create self.sortedcelldict here
        # key is cell id, [sortedclass id, node id, cropped img]
        celldict = {}
        count = 0
        for key,img in celldict_temp.items():
            k = np.int32(key)
            celldict[str(k)] = [nodedict[str(correctnode[count])], correctnode[count], img]
            count += 1
                    
        # create self.sortedclassdict here               
        classdict = {}
        for classid,cell_class in enumerate(self.cellclasses):
            classdict[str(classid)] = []
            for node in cell_class:
                for k in np.where(correctnode==node)[0]: #k is cell ID
                    key = int(list(celldict_temp.items())[k][0])                    
 
                    img = np.reshape(cellmat[:,k], (maxsize,maxsize))
                    classdict[str(classid)].append([node,k,img])

                    # feature 1 is 'count'
                    features_new[0,classid,0] += 1 
        
        self.sortedcelldict[path] = celldict
        self.sortedclassdict[path] = classdict
        self.sorteddataframe.append(features_new)
    #####################################################
    
    def extractfeatures(self,path):
        #Calculates cluster size, ratio of lenght to with, mean size of a cells in a class, mean variance of cells in a class
        #
        # Input:
        # - path - path of the image to process
        #Usage
        #Fvector = extractfeatures(path) #vector of features
        #####
        
        stats, cells = self.readimgandfilter(path)
        celldict = self.getsortedcelldict(path)
        
        stats["cell_class"] = np.nan
        
        #asign class to every cell in the stats
        for i, row in stats.iterrows():
            # If cell was filtered out, just continue
            if str(i) not in celldict:
                continue
            stats.loc[i, ["cell_class"]] = celldict[str(i)][0]
        
        #drop cell that have no class assigned by segimg
        if stats["cell_class"].isnull().sum() >= 1:
            stats.dropna()
        
        #compute features in a grouped level for every cell class
        grouped = stats.groupby("cell_class")
        count = grouped.size() #number of cells of each class on the picture
        stats["size_est"] = stats.loc[:,2] - stats.loc[:,3]
        size_est = grouped.mean().size_est
        mnsizes = grouped.mean()[4] #mean size in a group, probably will reflect the cell class
        sdsizes = grouped.std()[4] #sd in a group, not sure if this still makes sence
        
        slice2stage = {
        "417": "IDC",
        "418": "IDC",
        "419": "DCIS",
        "420": "DCIS",
        "421": "EN",
        "422": "EN",
        "423": "EN",
        "424": "normal",
        "425": "normal"
        }
        
        
        #prep your data frame
        
        from collections import OrderedDict 
        features = OrderedDict([
         ("count", count),
         ("size_est", size_est),
         ("mnsizes", mnsizes),
         ("sdsizes", sdsizes),
        ])
        
        celltypes = range(4)#don't know how to make this line check how long the cellclasses object is
        
        FeatDF = pd.DataFrame(0, index = celltypes, columns = features.keys())
        FeatDF["id"] = celltypes
        
        for col, feature in features.items():
            FeatDF.loc[feature.index,col] = feature
            
        melted = pd.melt(FeatDF, id_vars=['id'], value_vars=features.keys())
        
        pic = os.path.basename(path)
        stage = slice2stage[str(pic.split('_')[1])]
        Fvector = np.concatenate((pic,stage,melted.value),axis=None)
        
        return(Fvector)
    #####################################################
    def __del__(self):
        self.cellclass = []
        self.correctnodedict ={}
        self.cellimagedict = {}
        self.sortedcelldict={}
        self.sortedclassdict={}
        self.sorteddataframe=[]
        print("deleted")
        
    def delete_memory(self):
        self.cellclass = []
        self.correctnodedict ={}
        self.cellimagedict = {}
        self.sortedcelldict={}
        self.sortedclassdict={}
        self.sorteddataframe=[]
        print("deleted")
    ##################################################### 
    def printnodefig(self, path, avgnodes = False):
        # Prints a hexagonal grid with the average cell in each node
        #
        # Input:
        # - path - path of the image to process
        #####
        
        mapsize = np.int32(np.sqrt(self.SOMvecs.shape[1])) 
        mapsize = [mapsize, mapsize]
        cellmat = self.cellimagedict[path]
        cellclasses = self.cellclasses
        correctnode = self.correctnodedict[path]
        maxsize = self.maxsize
        features = self.getsorteddataframe()
        pathindex = [ind for ind, path_ in enumerate(self.pathlist) if path in path_]

        imgsize = [100,100]
        trans = [131,114]
        #trans = [131,114]

        x0_2 = [70,70]#[869,316];
        x0_1 = [x0_2[0]+66,x0_2[1]]

        hex_new = np.zeros((1000,1130,3))
        
        #Unpackage class dict
        classdict = {}
        numcellclasses = len(cellclasses)
            
        for k,cell_class in enumerate(cellclasses):
            for cell in cell_class:
                classdict[str(cell)] = k

        for i in range(mapsize[0]*mapsize[1]):
            j=i+1
                
            images = np.float32(np.reshape(cellmat[:,np.where(correctnode==i)],(maxsize,maxsize,-1)))
            num_in_class = images.shape[2]
            if avgnodes == True:
                gray = np.reshape(np.dot(self.PCAcoef,self.SOMvecs[:,i]),(maxsize,maxsize))
            else:
                gray = np.mean(images,axis=2)
                #gray = np.reshape(cellmat[:,np.where(correctnode==i)[0][0]],(maxsize,maxsize))
          
            #gray = np.reshape(cellmat[:,np.where(correctnode==i)[0][0]],(maxsize,maxsize))
           
            #rgb = cat(3, uint8(gray*255), zeros(size(gray)), zeros(size(gray)));
            rgb_ = cv2.resize(gray, dsize=(imgsize[0], imgsize[1]), interpolation=cv2.INTER_CUBIC)
            
            #if i==45: 
            #    plt.imshow(rgb_)
            
            #rgb transformation
            rgb_ /= np.amax(rgb_.astype(float),axis=(0,1))
            rgb_ = np.transpose(np.tile(rgb_, [3,1,1]), [1,2,0])
            
            if i==45:
            	scipy.io.savemat('modules/rgb_.mat', mdict={'rgb_': rgb_})            

            rgb_[rgb_<0] = 0
            #rgb_ = np.tile(rgb_, [1,1,3])
            #rgb_temp = np.zeros((rgb_.shape[0],rgb_.shape[1],3))
            #for i in range(3):
            #   rgb_temp[:,:,i] = rgb_
            #rgb_ = rgb_temp
            #print(rgb_.shape)
            #if i==45: 
            #    plt.imshow(rgb_)


            if str(i) in classdict.keys():
                color = colorsys.hsv_to_rgb(self.color_permute[classdict[str(i)]]/numcellclasses,1.,1.)
            else:
                color = [1.,1.,1.]
                
            color_ = np.tile(color, [rgb_.shape[0],rgb_.shape[1],1])
            #rgb_ = np.invert(np.uint8(rgb_*255  * color_))
            rgb_ = (np.uint8(rgb_ *255 *color_))
            
            
            factor = max(np.floor((j-1)/mapsize[0]),0)
            #print('factor',factor)
            remainder = j - factor * mapsize[0];

            if factor % 2 == 0:
                x0 = x0_1
            else:
                x0 = x0_2

            #ycoord = [x0[1]:(x0[1]+imgsize[0]-1)] + trans[1]*factor
            #xcoord = [x0[0]:(x0[0]+imgsize[1]-1)] + trans[0]*(remainder-1)
            ycoord = np.int32(np.array([x0[1],x0[1]+imgsize[0]]) + trans[1]*factor)
            xcoord = np.int32(np.array([x0[0],x0[0]+imgsize[1]]) + trans[0]*(remainder-1))
            #print(ycoord[0],ycoord[1])
            #print(xcoord[0],xcoord[1])
            #print('###############')
            #print(np.amax(rgb_))
            rgb_ = scipy.ndimage.rotate(rgb_, 90)
            hex_new[ycoord[0]:ycoord[1], xcoord[0]:xcoord[1],:] = rgb_
            
            #Write Text in Image
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (xcoord[0],ycoord[0])
            fontScale = 0.7
            fontColor = (1,1,1)
            lineType = 2
            cv2.putText(hex_new, str(i) + ': ' + str(num_in_class), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
              

        #print(df)
        plt.figure(figsize=(20,20))
        plt.title('User-defined cell clusters', fontsize=30)#, [Node ID,Number of Cells in Node]
        plt.tick_params(axis='both', labelsize = 32)
        plt.imshow((hex_new[:850,:1050,:]).astype(np.uint8))

        plt.figure(figsize=(20,20))
        #plt.title('Number of Cells in Each Node', fontsize=30)
        plt.tick_params(axis='both', labelsize = 32)
        plt.hist(correctnode,mapsize[0]*mapsize[1])
        
        print(np.squeeze(features[pathindex[0],:,0]).shape)
        plt.figure(figsize=(20,20))
        #plt.title('Class Histogram', fontsize=30)
        plt.tick_params(axis='both', labelsize = 32)
        plt.bar(np.arange(len(cellclasses)),np.squeeze(features[pathindex[0],:,0]))
        
        return(hex_new[:850,:1050,:].astype(np.uint8))
