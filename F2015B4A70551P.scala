package pplAssignment

object F2015B4A70551P{
    //Start Coding from here
    //----------------------------------------------Convolution -------------------------------------------
    //multiply 2 lists
    def mul(a:List[Double], b:List[Double]):Double = {  
      if(a.nonEmpty){
        a.head * b.head + mul(a.tail,b.tail)
      }
      else 0
     }
    //dot of 2 matrices
    def dotProduct(matrix_1:List[List[Double]],matrix_2:List[List[Double]]):Double = matrix_1 match{  
      case Nil => 0
      case h::t => matrix_2 match{
        case h2::t2 => mul(h,h2) + dotProduct(t,t2)
        case Nil => 0
      }
    }
    //From the matrix cuts out the required matrix of size K x J
    def boxs( Image:List[List[Double]],J:Int ,K:Int, skip:Int):List[List[Double]] = {
      def creates(l1:List[Double],k:Int,skip:Int):List[Double] = {
        if(skip != 0)
        {
          creates(drop(l1,1),k,skip-1)
        }
        else if(l1.nonEmpty && k>0 && skip == 0)
        {
          List(l1.head) ::: creates(l1.tail,k-1,0)
        }
        else Nil
      }
      if(Image.nonEmpty && J>0)
          List(creates(Image.head,K,skip)) ::: boxs(Image.tail,J-1,K,skip)
      else Nil 
    }
    // cuts of matrices and convolutes each with the kernel and returns the convoluted matrix
    def convolute(Image:List[List[Double]], Kernel:List[List[Double]], imageSize:List[Int], kernelSize:List[Int] ):List[List[Double]] = {
      val K = kernelSize.head
      val J = kernelSize.tail.head
      val r = imageSize.head
      val c = imageSize.tail.head
      def convoluteList(Image:List[List[Double]],skip:Int):List[Double] =
      {
    		if(skip<=r-K)
        {
          List(dotProduct(boxs(Image,J,K,skip),Kernel)) ::: convoluteList(Image,skip+1)
        }
        else Nil
      }
      def convoluteListList(Image:List[List[Double]],skip:Int):List[List[Double]] = 
      {
        if(skip<=c-J)
        {
          List(convoluteList(Image,0)) ::: convoluteListList(Image.tail,skip+1)
        }
        else Nil
      }
      convoluteListList(Image,0)
      
    }
    //-----------------------------------------------------Activation Layer-----------------------------------------------
    def activationLayer(activationFunc:Double => Double, Image:List[List[Double]]):List[List[Double]] = {
      def activateList(l1:List[Double]):List[Double] = {
        if(l1.nonEmpty)
        {
          List(activationFunc(l1.head)) ::: activateList(l1.tail)
        }
        else Nil
    	}
      if(Image.nonEmpty)
      {
        List(activateList(Image.head))::: activationLayer(activationFunc, Image.tail) 
      }
      else Nil   
    }
    //---------------------------------------------------ActivationFunctions----------------------------------------------
    def ReLu(x:Double):Double = {
    	if(x < 0) 0
    	else x
    }
    def LeakyReLu(x:Double):Double = {
    	if(x < 0) x/2
    	else x
    }
    //-------------------------------------------------------Pooling-------------------------------------------------------
    def box( Image:List[List[Double]],J:Int ,K:Int, skip:Int):List[Double] = {
      def create(l1:List[Double],k:Int,skip:Int):List[Double] = {
        if(skip != 0)
          create(drop(l1,k),k,skip-1)
        else if(l1.nonEmpty && k>0 && skip == 0)
                  List(l1.head) ::: create(l1.tail,k-1,0)
             else Nil
      }
      if(Image.nonEmpty && J>0)
        {
          create(Image.head,K,skip) ::: box(Image.tail,J-1,K,skip)
        }
        else Nil 
    }
    def singlePooling(poolingFunc:List[Double]=>Double, Image:List[List[Double]], K:Int): List[Double]=
    {
      def help(Image:List[List[Double]],K:Int, skip:Int,poolingFunc:List[Double]=>Double):List[Double] =
      {
      if(box(Image,K,K,skip).nonEmpty)
        List(poolingFunc(box(Image,K,K,skip))) ::: help(Image,K,skip+1,poolingFunc)
      else
        Nil
      }
      help(Image,K,0,poolingFunc)
    }
    def poolingLayer (poolingFunc:List[Double]=>Double, Image:List[List[Double]], K:Int):List[List[Double]] = {
      def shiftbot(Image:List[List[Double]] , K:Int):List[List[Double]] = {
        if(K != 0)
        	shiftbot(Image.tail,K-1)
        else
        	Image
      }
      if(Image.nonEmpty)
      List(singlePooling(poolingFunc,Image,K)) ::: poolingLayer(poolingFunc,shiftbot(Image,K),K)
      else
      Nil
    }
    //----------------------------------------------------Pooling functions-------------------------------------------------
    def max(l:List[Double]):Double = {
      maxList(l)
    }
    def avg(l:List[Double]):Double = {
      sumList(l)/lenList(l)
    }
    //---------------------------------------------------Helpers------------------------------------------------------------
    //skips first n element and returns new list
    def drop(l:List[Double],n:Int):List[Double] = l match{
      case Nil=>Nil
      case _ if n==0 => l
      case _::tail => drop(tail,n-1)
    }
    //returns sum of list
    def sumList(l:List[Double]):Double = {
      if(l.nonEmpty)
      {
        l.head + sumList(l.tail)
      }
      else {0}
    }
    //returns length of list
    def lenList(l:List[Double]):Int = {
      if(l.nonEmpty)
      {
        1 + lenList(l.tail)
      }
      else 0
    }
    //returns height of 2D list
    def heightList(l:List[List[Double]]):Int = {
      if(l.nonEmpty)
      {
        1 + heightList(l.tail)
      }
      else 0
    }
    //returns max element in list
    def maxList(l:List[Double]):Double = {
      def maxl(l:List[Double],ans:Double):Double = {
        if(l.nonEmpty && l.head > ans )
          maxl(l.tail,l.head)
        else if(l.nonEmpty && l.head <=ans )
          maxl(l.tail,ans)
        else ans
      }
      maxl(l,0)
    }
    //returns max element in list of lists
    def maxListList(l:List[List[Double]]):Double = {
      def maxll(l:List[List[Double]],ans:Double):Double = {
        if(l.nonEmpty && maxList(l.head) > ans)
        	maxll(l.tail,maxList(l.head))
        else if (l.nonEmpty && maxList(l.head) <= ans)
          maxll(l.tail,ans)
        else ans
      }
      maxll(l,0)
    }
    //returns min element in list
    def minList(l:List[Double]):Double = {
      def minl(l:List[Double],ans:Double):Double = {
        if(l.nonEmpty && l.head < ans )
          minl(l.tail,l.head)
        else if(l.nonEmpty && l.head >=ans )
          minl(l.tail,ans)
        else ans
      }
      minl(l,999999)
    }
    //returns min element in list of lists
    def minListList(l:List[List[Double]]):Double = {
      def minll(l:List[List[Double]],ans:Double):Double = {
        if(l.nonEmpty && minList(l.head) < ans)
        	minll(l.tail,minList(l.head))
        else if (l.nonEmpty && minList(l.head) >= ans)
          minll(l.tail,ans)
        else ans
      }
      minll(l,999999)
    }
    //---------------------------------------------------------Normalisation---------------------------------------------------
    //normalisation function
    def normalVal(n:Double, minn:Double, maxx:Double):Int = {
      (255*(n - minn)/(maxx - minn)).round.toInt
    }
    def normalise(Image:List[List[Double]]):List[List[Int]] = {
      val maxx = maxListList(Image)
    	val minn = minListList(Image)
      def normaliseList(l:List[Double]):List[Int] = {
        if(l.nonEmpty)
          {
            List(normalVal(l.head,minn,maxx) ) ::: normaliseList(l.tail)
          }
        else Nil
      }
      def normaliseListList(Image:List[List[Double]],minn:Double, maxx:Double):List[List[Int]] =
      {
      if(Image.nonEmpty)
    	List(normaliseList(Image.head)) ::: normaliseListList(Image.tail,minn,maxx)
      else Nil
      }
      normaliseListList(Image,minn,maxx)
    } 

    //------------------------------------------------------MixedLayer----------------------------------------------------------
    def mixedLayer(Image:List[List[Double]], Kernel:List[List[Double]], imageSize:List[Int], kernelSize:List[Int], activationFunc:Double => Double, poolingFunc:List[Double]=>Double,  K:Int):List[List[Double]] = {
      poolingLayer(poolingFunc,activationLayer(activationFunc,convolute(Image,Kernel,imageSize,kernelSize)),K)
    }
    //----------------------------------------------Assembly------------------------------------------------
    //multiply matrix with weight
    def weight(Image:List[List[Double]],weight:Double):List[List[Double]] = {
      def weightList(l:List[Double]):List[Double] = {
        if(l.nonEmpty)
            List( l.head * weight ) ::: weightList(l.tail)
        else Nil
      }
      def weightListList(Image:List[List[Double]]):List[List[Double]] = {
      if(Image.nonEmpty)
        List(weightList(Image.head)) ::: weightListList(Image.tail)
      else Nil
      }
      weightListList(Image)
    }
    //add 2 matrices with bias
    def add(Image1:List[List[Double]],Image2:List[List[Double]],bias:Double):List[List[Double]] = {
      def addList(l1:List[Double],l2:List[Double]):List[Double] = {
        if(l1.nonEmpty)
            List(l1.head + l2.head + bias) ::: addList(l1.tail,l2.tail)
        else Nil
      }
      def addListList(l1:List[List[Double]],l2:List[List[Double]]):List[List[Double]] = {
        if(l2.nonEmpty)
          List(addList(l1.head,l2.head)) ::: addListList(l1.tail,l2.tail)
        else Nil
      }
      addListList(Image1,Image2)
    }
    //find new size
    def newSize(l:List[List[Double]]):List[Int] = {
      List(lenList(l.head),heightList(l))
    }
    def assembly(Image:List[List[Double]], imageSize:List[Int], w1:Double, w2:Double, b:Double, 
                 Kernel1:List[List[Double]], kernelSize1:List[Int], Kernel2:List[List[Double]], kernelSize2:List[Int], Kernel3:List[List[Double]], kernelSize3:List[Int], Size: Int):List[List[Int]] = {
      val t1 = mixedLayer(Image,Kernel1,imageSize,kernelSize1,ReLu,avg,Size)
      val t2 = mixedLayer(Image,Kernel2,imageSize,kernelSize2,ReLu,avg,Size)
      val t3 = add(weight(t1,w1),weight(t2,w2),b)
      val t4 = mixedLayer(t3,Kernel3,newSize(t3),kernelSize3,LeakyReLu,max,Size)
      normalise(t4)
    }
}