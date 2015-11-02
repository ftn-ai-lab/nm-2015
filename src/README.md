## 01_mnist_mlp.py

### Zadatak

Eksperimentisati sa:

* strukturom neuronske mreže (broj skrivenih slojeva, broj neurona u skrivenom sloju)
* parametrima obučavanja (brzina obučavanja, momentum)
* aktivacionim funkcijama
* brojem elemenata koje SGD uzima u svakoj epohi
* brojem epoha

**Za svaki eksperiment zabeležiti sve gorenavedene parametre i postignutu tačnost**

------

## 02_cifar10_cnn.py

Implementirana CNN za CIFAR10 dataset. Moguće je koristiti težine već obučene CNN učitavanjem fajla *02_cifar10_cnn.hdf5*.

**Napomena:** Ukoliko budete menjali samu arhitekturu CNN, moraćete je ponovo obučiti, tj. neće biti moguće koristiti težine prethodno obučene CNN.

### Zadatak

CIFAR10 CNN radi samo sa 32x32 RGB slikama na kojima se nalazi samo po jedan objekat određene klase (ukupno 10 klasa).
U cilju prepoznavanja više objekata određene klase ili različitih klasa, potrebno je implementirati tzv. "heat-map", odnosno
da se na slici proizvoljne veličine mogu pronaći objekti.

Primer heat mape za klasu *dog* (crvenije/toplije nijanse su indikator većih verovatnoća sa određenu klasu):

<img src="images/heatmap.png" >

Krajni cilj je da, na osnovu dobijene heat mape, na originalnoj slici obeležiti prepoznate objekte.

** Testirati nad slikama ** "images/02_\*.jpg"

------
