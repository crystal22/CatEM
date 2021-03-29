后缀为_cell_poi 文件中，存储了poi的相关信息，其中包括poi的id，名字，经纬度，类别id，类别名称以及它所在的网格id。

后缀为_land2type(1)的文件，存储了相关网格的信息，包括经纬度、网格类别等等。

后缀为_cell_info的文件存储的是包含了poi的网格信息，即_land2type(1)中包含的网格数量更多。

删去不在网格内poi后，各城市 poi 各大类别数量如下：

![图片](https://user-images.githubusercontent.com/78341268/112838708-ed5f2e00-90cf-11eb-959a-0c86aed683b4.png)

其中，纽约的网格类别代码为：

1	Residential

2	Mixed Residential & Commercial Buildings

3	Commercial & Office Buildings

4	Transportation & Utility

5	Open Space & Outdoor Recreation

6	Other

其余四个城市的网格类别代码为：

1	High Density Urban Fabric

2	ML-Density Urban Fabric

3	Industrial,commercial, public, military and private units

4	Open Space& Recreation

5	Transportation

