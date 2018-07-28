介绍：
==
代码都是本人参考李纯明教授的matlab源码实现的。
做了一定的优化。
效果很好，能达到李纯明教授的matlab源码运行效果，且速度更快。

可以到[李纯明教授个人博客](http://www.engr.uconn.edu/~cmli/)下载matlab代码并对照此代码,算法的实现逻辑基本一致，可以在此代码上进行修改能做更多的优化和改进。

使用：
==
 >>1、配置opencv环境。（本人项目是在opencv3.4.1版本上运行的。应该版本3以上都可以）

>>2、如下三种初始化方式（有些差别，但很有用），参数可以试着调。

>>方式一、
<pre>
Mat input();                        //这里初始化一个要处理的图像。
Rect set0();                        //这里初始化一个矩形区域。
Drlse dr(1, 5, 5, -3, 1.5, 0.8, DOUBLEPOT);
dr.init(input, set0);               //传入待处理图像，并将矩形区域作为初始轮廓。
dr.evolution(80, true);             //传入迭代次数和是否显示迭代过程。
Mat output = dr.getres();           //ouput则为得到的最终轮廓。
</pre>
>>方式二、
<pre>
Mat input();                        //这里初始化一个要处理的图像。
Rect set0();                        //这里初始化一个矩形区域。
Rect roi();                         //这里初始化一个感兴趣区域。（作为一种优化）
Drlse dr(1, 5, 5, -3, 1.5, 0.8, DOUBLEPOT);
dr.init(input， roi, set0);         //传入待处理图像，并传入感兴趣区域，并将矩形区域作为初始轮廓。
dr.evolution(80, true);             //传入迭代次数和是否显示迭代过程。
Mat output = dr.getres();           //ouput则为得到的最终轮廓。
</pre>
>>方式三、
<pre>
Mat input();                        //这里初始化一个要处理的图像。
Mat set0();                         //这里初始化一个二值图像（必须是二值图）。
Rect roi();                         //这里初始化一个感兴趣区域。（作为一种优化）
Drlse dr(1, 5, 5, -3, 1.5, 0.8, DOUBLEPOT);
dr.init(input， roi, set0);         //传入待处理图像，并传入感兴趣区域，并将二值图作为初始轮廓。（默认会将255的值作为轮廓内区域）
dr.evolution(80, true);             //传入迭代次数和是否显示迭代过程。
Mat output = dr.getres();           //ouput则为得到的最终轮廓。
</pre>