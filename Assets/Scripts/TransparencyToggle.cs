using UnityEngine;
using Oculus.Interaction;
using Oculus.Interaction.HandGrab;

public class TransparencyToggle : MonoBehaviour
{
    [SerializeField]
    private Material sphereMaterial;
    
    [SerializeField]
    private float transparentAlpha = 0.4f;
    
    [SerializeField]
    private float opaqueAlpha = 1f;
    
    [SerializeField]
    private HandGrabInteractable handGrabInteractable;
    
    [SerializeField]
    private Color baseColor;
    
    [SerializeField]
    private MeshRenderer meshRenderer;
    
    void Start()
    {
        // Get the material if not assigned
        if (sphereMaterial == null)
        {
            if (meshRenderer != null)
            {
                sphereMaterial = meshRenderer.material;
            }
        }
        
        if (sphereMaterial != null)
        {
            baseColor = sphereMaterial.color;
        }
        
        // Subscribe to events
        if (handGrabInteractable != null)
        {
            handGrabInteractable.WhenPointerEventRaised += HandlePointerEvent;
        }
    }
    
    void OnDestroy()
    {
        // Unsubscribe from events
        if (handGrabInteractable != null)
        {
            handGrabInteractable.WhenPointerEventRaised -= HandlePointerEvent;
        }
    }
    
    private void HandlePointerEvent(PointerEvent pointerEvent)
    {
        if (sphereMaterial == null) return;
        
        switch (pointerEvent.Type)
        {
            case PointerEventType.Hover:
            case PointerEventType.Select:
                // Hand is touching/grabbing - make opaque
                Debug.Log("HAND IS TOUCHING!");
                SetAlpha(opaqueAlpha);
                break;
                
            case PointerEventType.Unhover:
            case PointerEventType.Unselect:
            case PointerEventType.Cancel:
                // Hand released - make transparent
                Debug.Log("HAND STOPPED TOUCHING!");
                SetAlpha(transparentAlpha);
                break;
        }
    }
    
    private void SetAlpha(float alpha)
    {
        if (sphereMaterial != null)
        {
            Color newColor = baseColor;
            newColor.a = alpha;
            sphereMaterial.color = newColor;
        }
    }
}
