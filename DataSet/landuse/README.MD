后缀为_cell_poi 文件中，存储了poi的相关信息，其中包括poi的id，名字，经纬度，类别id，类别名称以及它所在的网格id。

后缀为_land2type(1)的文件，存储了相关网格的信息，包括经纬度、网格类别等等。

后缀为_cell_info的文件存储的是包含了poi的网格信息，即_land2type(1)中包含的网格数量更多。

删去不在网格内poi后，各城市 poi 各大类别数量如下：

	NYC	Milano	Amsterdam	Barcelona
Food	48749	21242	12825	30532
Shop&Service	70405	27458	22254	34377
Outdoors&Recreation	14914	6385	8712	8052
Professional&OtherPlaces	78287	24100	26004	23938
Travel&Transport	13489	3875	5448	5808
Arts&Entertainment	11787	4139	3966	4920
NightlifeSpot	9304	4821	4147	7820
Residence	13748	1898	2318	2686
College&University	6956	2781	2422	3416
Event	20	2	19	9
all	267659	96701	88115	121558
![图片](https://user-images.githubusercontent.com/78341268/112826652-dca6bc00-90bf-11eb-86f7-2fb15c9a955f.png)



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

