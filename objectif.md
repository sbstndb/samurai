# objectif : 
    -   ajouter les add_point et add_interval sur les cellArray
    -   Mesurer les performances sur advection1D 

# fichiers : 
    -   algorithm/update/update_fiels:758/799
        --> Change the for_each_interval de CellList à CellArray

# Algorithm : 
   
Soit un std::vector<interval> vi ; 
On veut integrer les fonctions vi.add_point(i)  et vi.add_interval({start, end})

    -   Si vi vide : 
        --> vi.push_back(interval)
    -   Sinon
        -   Si vi.back().end == interval.start : 
            --> élargir interval via vi.back.end = interval.end
        -   Sinon si vi.back.erd < interval.start : 
            --> vi.push_back(intevral) 

Ensuite, une fois qu'on veut copier et modifier le CellArray raffiné en utilisant les mask de raffinement (Keep, Coarse, Refine), 

for 0 --> lmax

    -   If tag[i] == Refine
        --> cellarray[l+1].add_interval({2i, 2i+1})
    -   If tag[i] == Keep
        --> cellarray[l}.add_point(i) 
    -   If tag[i] == Coarsen 
        --> cellarray[l-1].add_point(i/2)



