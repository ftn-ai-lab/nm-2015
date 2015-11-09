## Uputsvo za instalaciju alata potrebnih za predmet Neuronske mreže (2015)


Alati koji će biti korišćeni na ovom kursu:

* **Anaconda (ver 2.3) - Python (ver 2.7)** distribucija sa preko 300 paketa za naučno istraživanje. Sadrži Python, PIP package manager i pomenute pakete/biblioteke.

* **OpenCV (ver 3.0.0)** - alati za računarsku viziju (eng. computer vision)  - **OPCIONO**

* **Theano (ver 0.7)** - Python biblioteka za optimizovanje simboličkih matematičkih izraza i numeričkih izračunavanja. 
Može da se izvršava na grafičkoj kartici (GPU) - CUDA, OpenCL...

* **Keras (ver 0.2)** - Python biblioteka za neuronske mreže, bazirana na Theano

Napomena: Biće prikazano uputstvo za instalaciju za Windows OS (ali i za Linux distribucije i Mac OSX je prilično slična instalacija).

----

### Instalacija - Anaconda


1. Preuzeti instalaciju za Anacondu sa [https://www.continuum.io/downloads](https://www.continuum.io/downloads). 
**OBAVEZNO: preuzeti verziju Anaconde sa Python-om 2.7 (a ne 3.4).**

2. Dupli-klik na preuzetu .exe datoteku i pratiti instrukcije za instalaciju.

3. Zapamtiti putanju gde je instalirana Anaconda (dalje u uputstvu ova putanja će se zvati ANACONDA_INSTALL_PATH)

----

### Instalacija - OpenCV (OPCIONO)


1. Preuzeti datoteku **opencv-3.0.0.exe** sa [http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.0.0/](http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.0.0/).

2. Dupli-klik na preuzetu .exe datoteku - ovo će zapravo samo otpakovati OpenCV na zadatu putanju.

3. Otići u direktorijum gde je otpakovan OpenCV i pronaći direktorijum **opencv/build/python/2.7**.

4. Kopirati datoteku **cv2.pyd** u direktorijum **ANACONDA_INSTALL_PATH/lib/site-packages**


### Instalacija - Theano

Prvo je neophodno instalirati određene "dependencies" za Theano.

* 1. Otvoriti **Command prompt** i uneti:
```code
conda install mingw libpython
```

* 2. i zatim:

```code
conda update conda
```

* 3. Sad još samo instalirati Theano sa PIP (*bleeding-edge* verzija):

```code
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
```

----

### Instalacija - Keras

1. Potrebno je samo instalirati Keras sa PIP, dakle otvoriti **Command prompt** i uneti:

```code
pip install keras
```

----

## Uputstvo za kreiranje virtualne mašine (VM)


Pored ručne instalacije svih potrebnih lata, moguće je koristiti i virtualnu mašinu
na kojoj su svi ovi alati već instalirani, ali je u pitanju Linux Mint 17.2 distribucija. 

VM možete preuzeti/kopirati sa svih računara u Park City učionicama. VM se nalazi na putanji "/home/student2014/VirtualBox VMs/AnacondaVM.7z".

### Instalacija virtualne mašine

1. Instalirati **Oracle VM VirtualBox (ver 5.x)**.

2. Preuzeti datoteku **AnacondaVM.7z** sa ... (biće dostupan link čim datoteka bude postavljena online).

3. Raspakovati datoteku **AnacondaVM.7z** -> dobiće se datoteka **AnacondaVM.vdi**.

4. Otvoriti VirtualBox i napraviti novu VM: New -> Name: AnacondaVM, Type: Linux, Version: Ubuntu (64-bit) -> Next.

5. Dodeliti bar 2GB (2048 MB) RAM za VM -> Next.

6. Izabrati **Use an existing virtual hard disk file** i sa diska odabrati datoteku **AnacondaVM.vdi** -> Create.

7. Pokrenuti AnacondaVM virtualnu mašinu.


### Ako VM ne može da se pokrene

1. Unutar **VirtualBox** -> desni klik na AnacondaVM -> Remove -> Remove only

2. Za svaki slučaj, pronaći gde se nalazi direktorijum **VirtualBox VMs** (u fajl sistemu) 
i ako u njemu postoji folder **AnacondaVM**, obrisati samo taj folder.

3. Uraditi sve od 4. koraka u **Instalacija virutalne mašine**
