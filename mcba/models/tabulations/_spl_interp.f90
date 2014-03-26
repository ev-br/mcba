!
! Evaluates the spline interpolation at a given point (see helpers.Tabulated class).
!
    real*8 function spl_interp_feval(x, xa, xb, fa, fb, fda, fdb)
    implicit none
    real*8, intent(in) :: x, xa, xb, fa, fb, fda, fdb
    real*8 :: dx, dxa, dxb, deriv, A, B, f

        dx = xb-xa
        dxa = x-xa
        dxb = xb-x

        deriv = (fa-fb)/dx
        A = -( fdb + deriv )
        B =  fda + deriv
        
        f = A*dxa + B*dxb        
        f = f* ( dxa*dxb/dx/dx )
        f = f+ fa*dxb/dx + fb*dxa/dx 
    
        spl_interp_feval =  f
    
    end function spl_interp_feval
    

    
