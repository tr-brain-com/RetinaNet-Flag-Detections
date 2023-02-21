Bu çalışma, RetinaNet ile bir object detection işlemini train ve predict aşaması da dahil tüm süreçleri ile açıklamaya yöneliktir. Burada bahsedilen bilgiler kısa bir açıklama olmakla birlikte daha detaylı bilgileri yazacağım makale de belirteceğim.




RetinaNet Nedir? Ne Değildir?

RetinaNet,diğer görüntü sınıflandırma ve görüntü algılama algoritmaları gibi belirli bir sorunun çözümü için geliştirilmiş bir algoritmadır. Nedir ne değildir olayına geçmeden önce bu algoritmanın ortaya çıkışına sebebiyet veren sorunlara bir göz atmak akıllıca olacaktır.  “one-stage detector” algoritmalarında aşırıcı derecede ön plan-arka plan dengesizliğinin tespiti, bu algoritmanın performansının  “two-stage detectors (iki kademeli detectörler)” algoritmalarının performansından daha düşük olmasının temel nedeni olduğu düşüncesini de beraberinde getirdi.

Tek aşamalı bir detectör olan  RetinaNet, odak  kaybını kullanarak (by using focal loss) “kolay” örneklere daha az, “zor” örneklere daha daha çok odaklanarak tahmin doğruluğunu arttırmayı amaçlar. Özellik çıkarımı için ResNet+FPN, sınıflandırma ve sınırlayıcı kutu regresyonu için iki özel ağ kullanarak iyi bilinen R-CNN modellerinden daha iyi performans gösterdiği tespit edilmiştir. 

Bu genel bilgilerden sonra one-stage-detectors ve two-stage-detectors yapıları hakkında da kısaca bilgi edinelim.

One-Stage-Detectors (Tek Aşamalı Detectörler) :  Tek aşamalı bir detectör, sinir ağından tek bir geçiş gerektirir ve tüm sınırlayıcı kutuları tek seferde tahmin eder. Bu yapısı itibariyle özellikle mobil cihazlar için çok daha hızlı ve uygun bir yapı sunar. Fakat gözden kaçırılmaması gereken durum bu yapıda nesneler sınıflandırılırken sınırlayıcı kutu regresyonları, bölge önerileri (ROI) vs kullanılmadan işlem doğrudan yapılır. Bu da beraberinde hız getirse de özellikle küçük nesneler için (ayrıntıda yapılan işlemle için) tahmin değerlerini düşürmektedir. Bu yapıya örnek olarak vereceğimiz en yaygın algoritmaları şu şekilde sıralayabiliriz : 
YOLO
SSD
SqueezeDet
DetectNet
RetinaNet

Two-Stage-Detectors (iki Aşamalı Dedektörler):
iki aşamalı dedektörlerde bölge tespiti ve nesne sınıflandırması yerine bölge önerisi ve iki aşamalı dedektörlerle sınıflandırma ve yerelleştirme yapılır. Bölge önerisi ise  Bölge Öneri Ağı adı verilen RPN(Region Proposal Network) kullanılarak yapılır. Burada RPN, genel inference sürecini ve hesaplama maliyetlerini azaltmak için “nereye” bakılacağına karar vermek için kullanılır. RPN, belirli bir bölgede daha fazla işlem yapılması gerekip gerekmediğini değerlendirmek için her yeri hızlı ve verimli bir şekilde tarar. Bu konu farklı bir yazıya konu olabilir şimdilik daha fazla giriş yapmayacağım. Burada detaylı ve güzel bir anlatım mevcut, bir göz atmak isteyebilirsiniz.

Bu yapı, sözde bölge teklifleri(RPN) oluşturur (ROI)  ve ardından bu bölgelerin her biri için ayrı bir tahmin yapar. Bu iyi çalışıyor ancak modelin algılama ve sınıflandırma bölümünün birden çok kez çalıştırılmasını gerektirdiğinden one-stage detectors yapısına göre daha yavaş çalışmaktadır. Bu yavaşlığı doğruluk oranında iyileştirmelere tolere edebilir. İki aşamalı dedektörler genellikle daha iyi doğruluğa ulaşır ancak tek aşamalı dedektörlerden daha yavaştır. Bu modele örnek olarak bütün R-CNN model ailesi (Fast R-CNN, Faster R-CNN, Mask R-CNN) örnek olarak verilebilir. Kısaca çalışması ise şu şekildedir:

Birinci aşama, bölge öneri ağı (RPN), aday nesne konumlarının sayısını küçük bir sayıya (örneğin 1-2k) indirerek , çoğu arka plan örneğini filtreler.
İkinci aşamada ise her aday nesne konumu için sınıflandırma yapılır. Böylece, ön plan ve arka plan arasında yönetilebilir bir sınıf dengesine ulaşılabilmektedir.

Şimdi biraz da RetinaNet’in dedektör yapısından bahselim.



Şekil 1.1 RetinaNet Detectör Mimarisi

ResNet: Derin özellikler çıkarımı için kullanılan omurga
Feature Pyramid Network (FPN): Tek bir çözünürlülü giriş görüntüsünden zengin, çok ölçekli bir özellik piramidi oluşturmak için Resnet üzerinde kullanılır. FPN, son teknoloji sonuçlara sahip iki aşamalı bir dedektördür. 

FPN, çok ölçeklidir, tüm ölçeklerde anlamsal olarak çok güçlüdür ve hesaplaması hızlıdır.

Buraya kadar alattıklarımızdan sonra RetinaNet diğer benzer One-Stage-Detectors yapılarından ayıran Focal Loss kavramına da kısaca değinelim.

Focal Loss (Odak Kaybı): 

Focal Loss işlevi, nesne algılama gibi görevlerde eğitim sırasında sınıf düzensizliğini giderir. Odak kaybı, öğrenmeyi zor yada yanlış sınıflandırmış örneklere odaklamak için Cross Entropy Loss kaybına müdahale eder. Doğru sınıfa olan güven arttıkça ölçeklendirme faktörünün sıfıra düştüğü dinamik olarak ölçeklendirilmiş bir Cross Entropy Loss dur. Daha geniş ifadeyle bu ölçeklendirme faktörü, eğitim sırasında kolay örneklerin katkısını otomatik olarak azaltabilir ve modeli hızla zor örneklere odaklayabilir. Bu sayede zor veyas yanlış sınıflandırılmış örneklere yoğunlaşma hızlı bir şekilde sağlanmış olacak ve modelin doğruluğu hız kaybı olmadan arttırılmış olacaktır.

Buraya kadar bahsetmek istediğim mimariler, kullandıkları yapılar ve teknikler hakkında genel bir bilgi vermek üzerineydi. Artık kendimize ait bir veri setini kullanarak RetinaNet ile bir model eğitelim ve çıkarımlar yapalım.
Hadi Başlayalım :)

Öncelikle ilk aşamada RetinaNet kütüphanesini çalışma klasörümüze indirelim

git clone https://github.com/fizyr/keras-retinanet.git

Daha sonra aşağıda ki komutları çalıştırarak önce keras dizinine geçiyoruz ve sonrasında ise geras için kurulumu başlatıyoruz.

cd keras-retinanet/
pip install .

Sistemimizde keras ile çalışmak için son bir adım kaldı. Bu adım ile mevcut c dosyalarının python tarafından kullanılması için bazı kurulumlar gerçekleşecektir. Bu adımı atlarsanız bir çok hata alırsınız. Karşılaşacağınız hatalardan birisi ise “TypeError: Cannot interpret 'tf.float32' as a data type” hatasıdır. O yüzden aşağıda ki kodu yine keras-retinanet dizini altında çalıştırmalısınız.

python setup.py build_ext --inplace

Artık sistemimiz hazır, keras’ı kullanabiliriz. Bu çalışma da ben daha önce üzerine çalışılmamış olan “resimler üzerinde ki bayrakların tespiti” konusunu seçiyorum. Bu uygulamayı burada uçtan uca gerçekleştireceğiz. Öncelikle verisetimizi oluşturuyoruz. Burada kullanılan ”flag_detect” veri seti google üzerinden elde edildi ve öncelikle yolo formatında kullanılacak şekilde düzenlendi. Fakat RetinaNet Pascal-VOC formatını kullandığı için öncelikle bu dönüşümü gerçekleştirmemiz gerekecek.




train komutunu konsoldan çalıştırdığınız zaman muhtemelen “ModuleNotFoundError: No module named 'keras_retinanet.utils.compute_overlap' hatayı alacaksınız. Bunun nedeni yukarıda ki bahsettiğimiz c kodlarının çalışması için gerekli çalıştırılması gereken dosyayı çalıştırma manızdır. 
keras-retinanet klasörüne giderek “python setup.py build_ext --inplace” komutunu tekrar çalıştırdığınız zaman sorun düzelecektir.



ValueError: invalid CSV annotations file: dataset/flags/maskDetectorData.csv: line 1: unknown class name: ' flag ' (classes: OrderedDict([('flag', 0)]))


Bu şekilde bir hata alırsanız bunun nedeni bir şekilde class belirtirken clasın yanında ki boşluklardır. Bunu ben core kodlarına girerek düzelttim. Şu adımları takip et : 
keras_retinanet/keras_retinanet/preprocessing/csv_generator.py dosyasını açın ve 96-97. satırda bulunan 
if class_name not in classes:   ifadesini
if str(class_name).strip() not in classes:    olarak değiştirin. Bu şekilde class ile ilgili gelen hataların sebebi budur. Tüm hatalarda str ve strip işlemini uygulayın.
Şimdi öncelikle veri kümesi oluşturma aşamasını es geçiyorum. Siz kendi veri kümenizi Pascal VOC formatına uygun olarak oluşturduktan sonra buradan işleme devam edebilirsiniz. Bu çalışma için veri kümesi flag_dataset/test ve flag_dataset/train olarak olarak oluşturdum. Her bir klasör içerisinde annotaions ve images diye iki klasör daha bulunmaktadır. Zaten Pascal VOC formatına uygun olarak oluşturduğunuz veri kümesi size sorun çıkarmayacaktır. Klasör yapınız aşağıdaki şekilde olmalıdır.


Devamı için lütfen BU bağlantıyı takip ediniz.

