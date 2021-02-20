/* 
* This file is part of Fieldosophy, a toolkit for random fields.
*
* Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>
*
* This Source Code is subject to the terms of the BSD 3-Clause License.
* If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.
*
*/

#ifndef MISC_H
#define MISC_H


#include <list>
#include <set>
#include <algorithm>


// Class for storing global variables lists with identifiers
template <typename T> class GlobalVariablesList
{
    public:
    
        // Function for storing a member
        unsigned int store( const T & pT )
        {
                // If list is empty
                if (mList.size() == 0)
                    mList.push_back( std::pair<unsigned int, T>(0, pT) );
                else
                    mList.push_back( std::pair<unsigned int, T>( mList.back().first+1, pT ) );
                
                return mList.back().first;  
        }
        
        // Function for loading a member
        T * load( const unsigned int pId )
        {
            // Find member
            for( auto iter = mList.begin(); iter != mList.end(); ++iter) 
                if (iter->first == pId)
                {
                    return &(iter->second);
                }
                
            return NULL;
        }
    
        // Function for erasing stored member
        int erase( const unsigned int pId )
        {
            // Find mesh
            for( auto iter = mList.begin(); iter != mList.end(); ++iter) 
                if (iter->first == pId)
                {
                    // Remove mesh from storage
                    mList.erase(iter);
                    
                    return 0;
                }
                
            return 1;
        }
    
    private:
        // list
        std::list< std::pair<unsigned int, T> > mList;
};





// Get set intersection
inline void misc_setIntersection( const std::set<unsigned int> & p1, const std::set<unsigned int> & p2, std::set<unsigned int> & pOutput)
{
    pOutput.clear();
    // Get intersection of sets
    std::set_intersection( p1.begin(), p1.end(), p2.begin(), p2.end(), 
        std::inserter( pOutput, pOutput.begin() ) );
    
    return;
}
// Return true if sets are sharing element
inline unsigned int misc_numSharedElements( const std::set<unsigned int> & p1, const std::set<unsigned int> & p2, std::set<unsigned int> & pTemp) 
{
    misc_setIntersection(p1, p2, pTemp); 
    return pTemp.size();
}




extern "C"
{       
    
    // local cross-correlation between two images
    int misc_localMaxCrossCorr2D(
        const double * const pImage1, const double * const pImage2, 
        const unsigned int pWidth, const unsigned int pHeight,
        const unsigned int pTemplateRadius, const unsigned int pSearchRadius,
        const unsigned int pTemplateSkip, const unsigned int pSearchSkip,
        const unsigned int pTemplateStart, const unsigned int pSearchStart,
        unsigned int * const pOutput, const bool * const pEstimInd = NULL, double * const pCrossCorr = NULL
        );
}





#endif // MISC_H